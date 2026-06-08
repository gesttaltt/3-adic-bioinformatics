# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Core visualization utilities.

This package provides:
- Figure creation with automatic theming
- Export utilities for multiple formats
- Annotation helpers for scientific figures
"""

from .annotations import (  # Statistical annotations; Zone annotations; Legend helpers; Arrow annotations; Scale indicators
    add_annotation_arrow,
    add_category_legend,
    add_colorbar_annotation,
    add_correlation_annotation,
    add_goldilocks_zones,
    add_pvalue_annotation,
    add_reference_region,
    add_scale_bar,
    add_significance_bracket,
    add_threshold_line,
    create_legend_handles,
)
from .base import (  # Figure creation; Axes utilities; Colorbar and legend
    add_colorbar,
    add_inset_axes,
    add_legend,
    add_panel_label,
    create_3d_figure,
    create_figure,
    create_panel_figure,
    create_pitch_figure,
    create_scientific_figure,
    despine,
    set_axis_style,
)
from .export import (
    figure_to_array,  # Export functions; Utilities
    figure_to_base64,
    get_figure_size_inches,
    save_figure,
    save_figure_batch,
    save_plotly_figure,
    save_presentation_figure,
    save_publication_figure,
    save_web_figure,
    set_figure_size_inches,
)

__all__ = [
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
    # Export functions
    "save_figure",
    "save_publication_figure",
    "save_presentation_figure",
    "save_web_figure",
    "save_figure_batch",
    "save_plotly_figure",
    "get_figure_size_inches",
    "set_figure_size_inches",
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
    "add_colorbar_annotation",
]
