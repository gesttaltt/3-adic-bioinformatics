# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Professional visualization module for Ternary VAE Bioinformatics.

This package provides a unified, professional-quality visualization system with:

**Themes & Styling**
- Scientific publication style (Nature, Science, IEEE compatible)
- Presentation/pitch style (bold, high-contrast)
- Dark theme for demos and digital displays
- Notebook-optimized theme for Jupyter

**Color Palettes**
- Semantic palettes for biological/medical contexts (risk, safety, pathways)
- Colorblind-friendly categorical palettes (Paul Tol, Tableau 10)
- Custom colormaps for continuous data

**Figure Creation**
- Factory functions with automatic theme application
- Support for 2D, 3D, and panel layouts
- Consistent sizing for publications and presentations

**Export**
- Multi-format export (PNG, SVG, PDF)
- Publication-quality settings (300 DPI)
- Web-optimized output

Quick Start
-----------
>>> from src.visualization import (
...     create_figure,
...     use_scientific_style,
...     SEMANTIC,
...     save_figure,
... )
>>>
>>> # Apply scientific styling
>>> use_scientific_style()
>>>
>>> # Create figure with automatic styling
>>> fig, ax = create_figure()
>>> ax.plot([1, 2, 3], [1, 4, 9], color=SEMANTIC.primary)
>>> ax.set_title("Example Plot")
>>>
>>> # Save in publication-quality formats
>>> save_figure(fig, "example", formats=["png", "svg", "pdf"])

For Presentations
-----------------
>>> from src.visualization import use_pitch_style, create_pitch_figure
>>>
>>> use_pitch_style()
>>> fig, ax = create_pitch_figure(figsize=(12, 8))
>>> # Bold, high-contrast styling applied automatically

Theme Context Manager
---------------------
>>> from src.visualization import theme_context, Theme, Context
>>>
>>> with theme_context(Theme.SCIENTIFIC, Context.PAPER):
...     fig, ax = create_figure()
...     # Scientific paper styling active within this block
...     save_figure(fig, "paper_figure")
>>> # Original matplotlib settings restored

See Also
--------
- :mod:`src.visualization.styles` - Color palettes and themes
- :mod:`src.visualization.core` - Figure creation and export
- :mod:`src.visualization.plots` - Specialized plot types
- :mod:`src.visualization.projections` - Mathematical projections
"""

from __future__ import annotations

__version__ = "0.1.0"

# Configuration
from .config import (  # Configuration classes; Configuration functions; Constants
    COLUMN_WIDTH_DOUBLE,
    COLUMN_WIDTH_SINGLE,
    DPI_PRESENTATION,
    DPI_PUBLICATION,
    GOLDEN_RATIO,
    Context,
    ExportFormat,
    Theme,
    VisualizationConfig,
    configure,
    get_config,
    set_config,
)

# Core functionality
from .core import (  # Figure creation; Axes utilities; Export; Annotations
    add_annotation_arrow,
    add_category_legend,
    add_colorbar,
    add_correlation_annotation,
    add_goldilocks_zones,
    add_inset_axes,
    add_legend,
    add_panel_label,
    add_pvalue_annotation,
    add_reference_region,
    add_scale_bar,
    add_significance_bracket,
    add_threshold_line,
    create_3d_figure,
    create_figure,
    create_legend_handles,
    create_panel_figure,
    create_pitch_figure,
    create_scientific_figure,
    despine,
    figure_to_array,
    figure_to_base64,
    save_figure,
    save_figure_batch,
    save_plotly_figure,
    save_presentation_figure,
    save_publication_figure,
    save_web_figure,
    set_axis_style,
)

# Styles
from .styles import (  # Palettes; Colormap functions; Color utilities; Theme functions
    NEUTRALS,
    PALETTE,
    SEMANTIC,
    STRUCTURE_COLORS,
    TABLEAU10,
    TOLMUTED,
    TOLVIBRANT,
    VAE_COLORS,
    apply_theme,
    color_gradient,
    darken,
    get_categorical_cmap,
    get_diverging_cmap,
    get_goldilocks_cmap,
    get_risk_cmap,
    get_safety_cmap,
    get_sequential_cmap,
    lighten,
    reset_theme,
    setup_poster_style,
    setup_publication_style,
    setup_slide_style,
    theme_context,
    use_dark_style,
    use_notebook_style,
    use_pitch_style,
    use_scientific_style,
    with_alpha,
)

__all__ = [
    # Version
    "__version__",
    # Configuration
    "VisualizationConfig",
    "Theme",
    "Context",
    "ExportFormat",
    "get_config",
    "set_config",
    "configure",
    "COLUMN_WIDTH_SINGLE",
    "COLUMN_WIDTH_DOUBLE",
    "GOLDEN_RATIO",
    "DPI_PUBLICATION",
    "DPI_PRESENTATION",
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
    # Figure creation
    "create_figure",
    "create_scientific_figure",
    "create_pitch_figure",
    "create_3d_figure",
    "create_panel_figure",
    # Axes utilities
    "despine",
    "set_axis_style",
    "add_panel_label",
    "add_inset_axes",
    "add_colorbar",
    "add_legend",
    # Export
    "save_figure",
    "save_publication_figure",
    "save_presentation_figure",
    "save_web_figure",
    "save_figure_batch",
    "save_plotly_figure",
    "figure_to_array",
    "figure_to_base64",
    # Annotations
    "add_significance_bracket",
    "add_pvalue_annotation",
    "add_correlation_annotation",
    "add_goldilocks_zones",
    "add_threshold_line",
    "add_reference_region",
    "create_legend_handles",
    "add_category_legend",
    "add_annotation_arrow",
    "add_scale_bar",
]
