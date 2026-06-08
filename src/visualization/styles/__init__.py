# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Visualization styles subpackage.

This package provides:
- Color palettes (semantic, categorical, colormaps)
- Theme configurations (scientific, pitch, dark, notebook)
- Style presets for common use cases
"""

from .palettes import (  # Color palette objects; Categorical palettes; Colormap functions; Color utilities
    NEUTRALS,
    PALETTE,
    SEMANTIC,
    STRUCTURE_COLORS,
    TABLEAU10,
    TOLMUTED,
    TOLVIBRANT,
    VAE_COLORS,
    color_gradient,
    darken,
    get_categorical_cmap,
    get_diverging_cmap,
    get_goldilocks_cmap,
    get_risk_cmap,
    get_safety_cmap,
    get_sequential_cmap,
    lighten,
    register_colormaps,
    with_alpha,
)
from .themes import (  # Theme application; Quick style functions; Preset functions
    apply_theme,
    reset_theme,
    setup_poster_style,
    setup_publication_style,
    setup_slide_style,
    theme_context,
    use_dark_style,
    use_notebook_style,
    use_pitch_style,
    use_scientific_style,
)

__all__ = [
    # Palettes
    "PALETTE",
    "SEMANTIC",
    "NEUTRALS",
    "TOLVIBRANT",
    "TOLMUTED",
    "TABLEAU10",
    "VAE_COLORS",
    "STRUCTURE_COLORS",
    # Colormap functions
    "get_risk_cmap",
    "get_safety_cmap",
    "get_goldilocks_cmap",
    "get_diverging_cmap",
    "get_sequential_cmap",
    "get_categorical_cmap",
    # Color utilities
    "lighten",
    "darken",
    "with_alpha",
    "color_gradient",
    "register_colormaps",
    # Theme functions
    "apply_theme",
    "reset_theme",
    "theme_context",
    "use_scientific_style",
    "use_pitch_style",
    "use_dark_style",
    "use_notebook_style",
    "setup_publication_style",
    "setup_poster_style",
    "setup_slide_style",
]
