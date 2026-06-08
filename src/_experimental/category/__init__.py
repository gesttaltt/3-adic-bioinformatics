# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Categorical Mathematics Module.

Implements category-theoretic structures for biological modeling:
- Sheaves: Local-to-global data consistency
- Functors: Structure-preserving maps between categories
- Natural Transformations: Maps between functors

Key Applications:
1. Sheaf Theory for Proteins:
   - Local structure (residue properties) -> Global structure (fold)
   - Consistency constraints across overlapping domains
   - Gluing conditions for protein assembly

2. Categorical Semantics:
   - Functors map between different representations
   - Natural transformations capture relationships
   - Limits/colimits for combining data

Mathematical Framework:
A presheaf F on a topological space X assigns:
- To each open set U, an object F(U)
- To each inclusion U ⊆ V, a restriction map F(V) -> F(U)

A sheaf additionally satisfies:
- Locality: If s,t ∈ F(U) agree on a cover, s = t
- Gluing: Compatible local sections glue to global section

References:
- MacLane & Moerdijk (1992): Sheaves in Geometry and Logic
- Robinson (2014): Topological Signal Processing
- Curry (2014): Sheaves, Cosheaves and Applications
"""

from src._experimental.category.functors import (
    CategoricalFunctor,
    CodonToProteinFunctor,
    LatentSpaceFunctor,
    NaturalTransformation,
    SequenceCategory,
)
from src._experimental.category.sheaves import (
    ProteinSheaf,
    ResidueSection,
    SheafConstraint,
    SheafGluing,
    SheafMorphism,
)

__all__ = [
    # Sheaves
    "ProteinSheaf",
    "SheafConstraint",
    "SheafGluing",
    "SheafMorphism",
    "ResidueSection",
    # Functors
    "CategoricalFunctor",
    "NaturalTransformation",
    "SequenceCategory",
    "CodonToProteinFunctor",
    "LatentSpaceFunctor",
]
