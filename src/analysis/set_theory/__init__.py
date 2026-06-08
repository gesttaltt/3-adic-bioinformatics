# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Set-theoretic analysis for mutation and resistance patterns.

Provides formal set theory operations for analyzing:
- Mutation sets and their relationships
- Cross-resistance patterns via set intersection
- Minimal resistance-conferring mutation sets
- Uncertainty handling via rough sets
- Hierarchical resistance via lattice structures

Key components:
- MutationSet: Formal set representation of mutations
- ResistanceProfile: Set-based resistance modeling
- RoughSet: Handle uncertain/incomplete data
- ResistanceLattice: Hierarchical resistance structure
- FormalConceptAnalysis: Genotype-phenotype relationships
"""

from src.analysis.set_theory.formal_concepts import (
    ConceptLattice,
    FormalConcept,
    FormalContext,
    ImplicationRule,
)
from src.analysis.set_theory.lattice import (
    LatticeNode,
    ResistanceLattice,
    ResistanceLevel,
)
from src.analysis.set_theory.mutation_sets import (
    CrossResistanceAnalyzer,
    Mutation,
    MutationSet,
    MutationSetAlgebra,
    ResistanceProfile,
)
from src.analysis.set_theory.rough_sets import (
    ApproximationSpace,
    RoughClassifier,
    RoughSet,
)

__all__ = [
    # Mutation sets
    "Mutation",
    "MutationSet",
    "ResistanceProfile",
    "MutationSetAlgebra",
    "CrossResistanceAnalyzer",
    # Rough sets
    "RoughSet",
    "RoughClassifier",
    "ApproximationSpace",
    # Lattice
    "ResistanceLattice",
    "LatticeNode",
    "ResistanceLevel",
    # Formal concepts
    "FormalContext",
    "FormalConcept",
    "ConceptLattice",
    "ImplicationRule",
]
