# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""HIV analysis visualization module.

Provides publication-quality visualizations for:
- Drug resistance landscapes
- CTL escape trajectories
- Antibody neutralization patterns
- Integrated constraint maps
"""

from .escape_plots import (
    plot_epitope_conservation,
    plot_hla_escape_landscape,
    plot_protein_escape_velocity,
)
from .integration_plots import (
    plot_constraint_landscape,
    plot_tradeoff_map,
    plot_vaccine_targets,
)
from .neutralization_plots import (
    plot_antibody_clusters,
    plot_bnab_sensitivity,
    plot_breadth_potency,
)
from .resistance_plots import (
    plot_cross_resistance_heatmap,
    plot_drug_class_embeddings,
    plot_mutation_classification,
    plot_resistance_correlation,
)

__all__ = [
    # Resistance plots
    "plot_resistance_correlation",
    "plot_mutation_classification",
    "plot_cross_resistance_heatmap",
    "plot_drug_class_embeddings",
    # Escape plots
    "plot_hla_escape_landscape",
    "plot_protein_escape_velocity",
    "plot_epitope_conservation",
    # Neutralization plots
    "plot_bnab_sensitivity",
    "plot_breadth_potency",
    "plot_antibody_clusters",
    # Integration plots
    "plot_constraint_landscape",
    "plot_vaccine_targets",
    "plot_tradeoff_map",
]
