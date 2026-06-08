# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Plot type implementations.

This package provides specialized plotting functions for:
- Manifold visualizations (latent space, embeddings)
- Surface plots (loss landscapes, 3D surfaces)
- Distribution plots (histograms, KDE, violin)
- Training metrics (loss curves, accuracy over time)
"""

from .manifold import (
    plot_distance_heatmap,
    plot_embedding_clusters,
    plot_geodesics,
    plot_poincare_disk,
    plot_radial_distribution,
)
from .training import (
    create_training_dashboard,
    plot_gradient_norms,
    plot_learning_rate,
    plot_loss_components,
    plot_parameter_histogram,
    plot_train_val_comparison,
    plot_training_curves,
)

__all__ = [
    # Manifold plots
    "plot_poincare_disk",
    "plot_geodesics",
    "plot_embedding_clusters",
    "plot_distance_heatmap",
    "plot_radial_distribution",
    # Training plots
    "plot_training_curves",
    "plot_loss_components",
    "plot_gradient_norms",
    "plot_learning_rate",
    "plot_train_val_comparison",
    "plot_parameter_histogram",
    "create_training_dashboard",
]
