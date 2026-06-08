# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Rough set theory for handling uncertainty in resistance classification.

Rough sets provide a mathematical framework for dealing with
incomplete or uncertain information by defining:
- Lower approximation: Certainly in the set
- Upper approximation: Possibly in the set
- Boundary region: Uncertain cases

This is particularly useful for drug resistance where:
- Some mutations definitively confer resistance
- Some mutations may or may not confer resistance
- The relationship depends on context (other mutations, lineage, etc.)

References:
    - Pawlak (1982): Rough Sets
    - Pawlak & Skowron (2007): Rudiments of Rough Sets
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from src.analysis.set_theory.mutation_sets import Mutation, MutationSet

T = TypeVar("T")


@dataclass
class ApproximationSpace(Generic[T]):
    """Approximation space defined by an equivalence relation.

    An approximation space (U, R) consists of:
    - U: Universe of objects
    - R: Equivalence relation partitioning U

    Objects are indiscernible if they belong to the same equivalence class.
    """

    universe: set[T]
    equivalence_classes: list[frozenset[T]]

    @classmethod
    def from_attributes(
        cls,
        objects: list[T],
        attribute_fn: Callable[[T], Any],
    ) -> ApproximationSpace[T]:
        """Create approximation space from attribute function.

        Objects with the same attribute value are indiscernible.

        Args:
            objects: List of objects
            attribute_fn: Function extracting attribute from object

        Returns:
            Approximation space
        """
        # Group by attribute value
        groups: dict[Any, list[T]] = {}
        for obj in objects:
            attr = attribute_fn(obj)
            if attr not in groups:
                groups[attr] = []
            groups[attr].append(obj)

        equivalence_classes = [frozenset(g) for g in groups.values()]

        return cls(
            universe=set(objects),
            equivalence_classes=equivalence_classes,
        )

    @classmethod
    def from_mutation_genes(
        cls,
        mutations: list[Mutation],
    ) -> ApproximationSpace[Mutation]:
        """Create space where mutations in same gene are indiscernible.

        Args:
            mutations: List of mutations

        Returns:
            Approximation space partitioned by gene
        """
        return cls.from_attributes(mutations, lambda m: m.gene)

    def get_equivalence_class(self, obj: T) -> frozenset[T]:
        """Get equivalence class containing object.

        Args:
            obj: Object to find

        Returns:
            Equivalence class (or empty if not found)
        """
        for ec in self.equivalence_classes:
            if obj in ec:
                return ec
        return frozenset()


class RoughSet(Generic[T]):
    """Rough set with lower and upper approximations.

    A rough set is defined by:
    - Lower approximation (certain members)
    - Upper approximation (possible members)
    - Boundary region = Upper - Lower (uncertain)

    Example:
        >>> # Mutations definitely conferring resistance
        >>> lower = MutationSet.from_strings(["rpoB_S450L"])
        >>> # Mutations possibly conferring resistance
        >>> upper = MutationSet.from_strings(["rpoB_S450L", "rpoB_H445Y", "rpoB_D435V"])
        >>> rough = RoughSet(lower, upper)
        >>> print(rough.boundary())  # {"rpoB_H445Y", "rpoB_D435V"}
    """

    def __init__(
        self,
        lower: set[T],
        upper: set[T],
        name: str = "",
    ):
        """Initialize rough set.

        Args:
            lower: Lower approximation (certainly in set)
            upper: Upper approximation (possibly in set)
            name: Optional name

        Raises:
            ValueError: If lower is not subset of upper
        """
        if not lower.issubset(upper):
            raise ValueError("Lower approximation must be subset of upper")

        self.lower = frozenset(lower)
        self.upper = frozenset(upper)
        self.name = name

    @classmethod
    def from_approximation_space(
        cls,
        target: set[T],
        space: ApproximationSpace[T],
        name: str = "",
    ) -> RoughSet[T]:
        """Create rough set from approximation space.

        Lower approximation: Union of equivalence classes contained in target
        Upper approximation: Union of equivalence classes overlapping target

        Args:
            target: Target set to approximate
            space: Approximation space
            name: Set name

        Returns:
            Rough set approximating target
        """
        lower: set[T] = set()
        upper: set[T] = set()

        for ec in space.equivalence_classes:
            if ec.issubset(target):
                # Entire class is in target -> lower approximation
                lower.update(ec)
                upper.update(ec)
            elif not ec.isdisjoint(target):
                # Class overlaps target -> upper approximation only
                upper.update(ec)

        return cls(lower, upper, name)

    @property
    def boundary(self) -> frozenset[T]:
        """Boundary region: Upper - Lower (uncertain members)."""
        return self.upper - self.lower

    @property
    def is_crisp(self) -> bool:
        """Check if set is crisp (no boundary)."""
        return self.lower == self.upper

    @property
    def roughness(self) -> float:
        """Roughness measure: 1 - |Lower| / |Upper|

        0 = crisp set, approaches 1 = very rough
        """
        if len(self.upper) == 0:
            return 0.0
        return 1 - len(self.lower) / len(self.upper)

    @property
    def accuracy(self) -> float:
        """Accuracy of approximation: |Lower| / |Upper|

        1 = crisp set, approaches 0 = very rough
        """
        return 1 - self.roughness

    def definitely_in(self, element: T) -> bool:
        """Check if element is definitely in set."""
        return element in self.lower

    def possibly_in(self, element: T) -> bool:
        """Check if element is possibly in set."""
        return element in self.upper

    def uncertain(self, element: T) -> bool:
        """Check if element is in boundary (uncertain)."""
        return element in self.boundary

    def __repr__(self) -> str:
        return f"RoughSet({self.name}, lower={len(self.lower)}, upper={len(self.upper)}, boundary={len(self.boundary)})"

    # Rough set operations

    def __and__(self, other: RoughSet[T]) -> RoughSet[T]:
        """Rough intersection."""
        return RoughSet(
            self.lower & other.lower,
            self.upper & other.upper,
        )

    def __or__(self, other: RoughSet[T]) -> RoughSet[T]:
        """Rough union."""
        return RoughSet(
            self.lower | other.lower,
            self.upper | other.upper,
        )

    def complement(self, universe: set[T]) -> RoughSet[T]:
        """Rough complement.

        ~Lower = U - Upper
        ~Upper = U - Lower
        """
        return RoughSet(
            universe - self.upper,
            universe - self.lower,
        )


@dataclass
class RoughClassifier:
    """Classifier based on rough set theory.

    Handles uncertainty by providing three-way classification:
    - Positive region: Definitely in class
    - Negative region: Definitely not in class
    - Boundary region: Uncertain, needs more information

    This is ideal for drug resistance where evidence may be incomplete.
    """

    positive_mutations: RoughSet[Mutation]
    name: str = ""

    @classmethod
    def from_evidence(
        cls,
        definite_resistance: list[str],
        possible_resistance: list[str],
        drug_name: str = "",
    ) -> RoughClassifier:
        """Create classifier from mutation evidence.

        Args:
            definite_resistance: Mutations definitely conferring resistance
            possible_resistance: Mutations possibly conferring resistance
            drug_name: Drug name for this classifier

        Returns:
            Rough classifier
        """
        lower = set(Mutation.from_string(m) for m in definite_resistance)
        upper = lower | set(Mutation.from_string(m) for m in possible_resistance)

        rough = RoughSet(lower, upper, f"{drug_name}_resistance")

        return cls(positive_mutations=rough, name=drug_name)

    def classify(self, mutations: MutationSet) -> str:
        """Classify a mutation profile.

        Args:
            mutations: Observed mutations

        Returns:
            Classification: "resistant", "susceptible", or "uncertain"
        """
        # Check for definite resistance mutations
        has_definite = any(mut in self.positive_mutations.lower for mut in mutations)

        if has_definite:
            return "resistant"

        # Check for possible resistance mutations
        has_possible = any(mut in self.positive_mutations.upper for mut in mutations)

        if has_possible:
            return "uncertain"

        return "susceptible"

    def classify_detailed(
        self,
        mutations: MutationSet,
    ) -> dict[str, Any]:
        """Detailed classification with evidence.

        Args:
            mutations: Observed mutations

        Returns:
            Detailed classification result
        """
        definite_hits = [m for m in mutations if m in self.positive_mutations.lower]
        possible_hits = [m for m in mutations if m in self.positive_mutations.boundary]

        classification = self.classify(mutations)

        confidence = self._compute_confidence(definite_hits, possible_hits)

        return {
            "classification": classification,
            "confidence": confidence,
            "definite_resistance_mutations": [str(m) for m in definite_hits],
            "possible_resistance_mutations": [str(m) for m in possible_hits],
            "evidence_strength": "strong" if definite_hits else ("moderate" if possible_hits else "none"),
        }

    def _compute_confidence(
        self,
        definite: list[Mutation],
        possible: list[Mutation],
    ) -> float:
        """Compute classification confidence.

        Args:
            definite: Definite resistance mutations found
            possible: Possible resistance mutations found

        Returns:
            Confidence score in [0, 1]
        """
        if definite:
            # High confidence with definite evidence
            return 0.9 + 0.1 * min(len(definite), 3) / 3
        elif possible:
            # Moderate confidence with possible evidence
            return 0.5 + 0.3 * min(len(possible), 3) / 3
        else:
            # High confidence in susceptibility
            return 0.85


@dataclass
class VariablePrecisionRoughSet(Generic[T]):
    """Variable Precision Rough Set (VPRS).

    Extends rough sets to allow some misclassification,
    controlled by precision parameters.

    This handles noisy biological data where strict
    set inclusion may be too restrictive.

    References:
        - Ziarko (1993): Variable Precision Rough Set Model
    """

    lower_threshold: float  # Inclusion threshold for lower (e.g., 0.9)
    upper_threshold: float  # Inclusion threshold for upper (e.g., 0.1)
    lower: frozenset[T] = field(default_factory=frozenset)
    upper: frozenset[T] = field(default_factory=frozenset)

    @classmethod
    def from_approximation_space(
        cls,
        target: set[T],
        space: ApproximationSpace[T],
        lower_threshold: float = 0.9,
        upper_threshold: float = 0.1,
    ) -> VariablePrecisionRoughSet[T]:
        """Create VPRS from approximation space.

        An equivalence class is in:
        - Lower if >= lower_threshold of class is in target
        - Upper if >= upper_threshold of class is in target

        Args:
            target: Target set
            space: Approximation space
            lower_threshold: Threshold for lower approximation
            upper_threshold: Threshold for upper approximation

        Returns:
            Variable precision rough set
        """
        lower: set[T] = set()
        upper: set[T] = set()

        for ec in space.equivalence_classes:
            overlap = len(ec & target)
            precision = overlap / len(ec) if ec else 0

            if precision >= lower_threshold:
                lower.update(ec)
                upper.update(ec)
            elif precision >= upper_threshold:
                upper.update(ec)

        return cls(
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
            lower=frozenset(lower),
            upper=frozenset(upper),
        )


class DecisionTable:
    """Decision table for rough set-based rules.

    Maps condition attributes to decision attributes,
    used for rule induction.
    """

    def __init__(
        self,
        objects: list[dict[str, Any]],
        condition_attrs: list[str],
        decision_attr: str,
    ):
        """Initialize decision table.

        Args:
            objects: List of objects (dictionaries)
            condition_attrs: Names of condition attributes
            decision_attr: Name of decision attribute
        """
        self.objects = objects
        self.condition_attrs = condition_attrs
        self.decision_attr = decision_attr

    def indiscernibility_classes(
        self,
        attributes: list[str] | None = None,
    ) -> list[list[int]]:
        """Compute indiscernibility classes.

        Args:
            attributes: Attributes to use (None = all condition)

        Returns:
            List of object index lists (classes)
        """
        attrs = attributes or self.condition_attrs

        # Group by attribute values
        groups: dict[tuple, list[int]] = {}
        for i, obj in enumerate(self.objects):
            key = tuple(obj.get(a) for a in attrs)
            if key not in groups:
                groups[key] = []
            groups[key].append(i)

        return list(groups.values())

    def lower_approximation(
        self,
        decision_value: Any,
    ) -> set[int]:
        """Compute lower approximation of decision class.

        Args:
            decision_value: Target decision value

        Returns:
            Set of object indices in lower approximation
        """
        # Get decision class
        decision_class = {i for i, obj in enumerate(self.objects) if obj.get(self.decision_attr) == decision_value}

        # Get indiscernibility classes
        ind_classes = self.indiscernibility_classes()

        # Lower = union of classes contained in decision class
        lower: set[int] = set()
        for cls in ind_classes:
            if set(cls).issubset(decision_class):
                lower.update(cls)

        return lower

    def upper_approximation(
        self,
        decision_value: Any,
    ) -> set[int]:
        """Compute upper approximation of decision class.

        Args:
            decision_value: Target decision value

        Returns:
            Set of object indices in upper approximation
        """
        decision_class = {i for i, obj in enumerate(self.objects) if obj.get(self.decision_attr) == decision_value}

        ind_classes = self.indiscernibility_classes()

        # Upper = union of classes overlapping decision class
        upper: set[int] = set()
        for cls in ind_classes:
            if not set(cls).isdisjoint(decision_class):
                upper.update(cls)

        return upper

    def quality_of_approximation(self) -> float:
        """Compute quality of approximation (γ).

        Measures how well condition attributes approximate decision.

        Returns:
            Quality in [0, 1]
        """
        # Get all decision values
        decision_values = set(obj.get(self.decision_attr) for obj in self.objects)

        # Compute positive region (union of lower approximations)
        positive_region: set[int] = set()
        for dv in decision_values:
            positive_region.update(self.lower_approximation(dv))

        return len(positive_region) / len(self.objects)

    def find_reducts(self) -> list[list[str]]:
        """Find reducts (minimal attribute subsets preserving quality).

        Returns:
            List of minimal attribute subsets
        """
        from itertools import combinations

        full_quality = self.quality_of_approximation()
        n_attrs = len(self.condition_attrs)

        reducts = []

        # Check subsets in order of increasing size
        for size in range(1, n_attrs + 1):
            for subset in combinations(self.condition_attrs, size):
                subset_list = list(subset)

                # Temporarily modify to use subset
                original_attrs = self.condition_attrs
                self.condition_attrs = subset_list

                quality = self.quality_of_approximation()

                self.condition_attrs = original_attrs

                if quality == full_quality:
                    # Check if minimal (no proper subset is also a reduct)
                    is_minimal = True
                    for prev_reduct in reducts:
                        if set(prev_reduct).issubset(set(subset_list)):
                            is_minimal = False
                            break

                    if is_minimal:
                        reducts.append(subset_list)

        return reducts
