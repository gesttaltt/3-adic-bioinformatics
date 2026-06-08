# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Mathematical projection visualizations.

This package provides specialized projections for:
- Poincare ball (hyperbolic geometry)
- Calabi-Yau surfaces
- Hopf fibration
"""

from .poincare import (
    compute_hyperbolic_centroids,
    create_poincare_animation_data,
    plot_3d_poincare_ball,
    plot_hierarchy_tree,
    project_to_2d_poincare,
)

__all__ = [
    "project_to_2d_poincare",
    "plot_3d_poincare_ball",
    "plot_hierarchy_tree",
    "compute_hyperbolic_centroids",
    "create_poincare_animation_data",
]
