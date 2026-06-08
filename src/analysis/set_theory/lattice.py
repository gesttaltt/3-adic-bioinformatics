# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Lattice structures for hierarchical resistance modeling.

The power set of mutations forms a Boolean lattice under subset ordering,
which naturally models resistance severity hierarchies:

    {all mutations} = XDR (top, ⊤)
         /    |    \\
       ...   MDR   ...
      /   \\  |  /   \\
    INH   RIF  FQ   AMK   = mono-resistant
      \\   /  |  \\   /
         ∅              = susceptible (bottom, ⊥)

Lattice operations:
- Meet (∧): Greatest lower bound (intersection)
- Join (∨): Least upper bound (union)

This structure enables:
- Ordering of resistance severity
- Finding common ancestors (shared mechanisms)
- Identifying minimal resistance elements
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Generic, TypeVar

from src.analysis.set_theory.mutation_sets import Mutation, MutationSet

T = TypeVar("T")


class ResistanceLevel(IntEnum):
    """Enumerated resistance levels with natural ordering."""

    SUSCEPTIBLE = 0
    MONO_RESISTANT = 1
    POLY_RESISTANT = 2  # Resistant to multiple first-line drugs
    MDR = 3  # Multi-drug resistant (INH + RIF)
    PRE_XDR = 4  # MDR + FQ or injectable
    XDR = 5  # MDR + FQ + injectable


@dataclass
class LatticeNode(Generic[T]):
    """Node in a lattice structure.

    Attributes:
        element: The element at this node
        level: Depth/level in lattice (0 = bottom)
        parents: Immediate predecessors (covers)
        children: Immediate successors (covered by)
    """

    element: T
    level: int = 0
    parents: list[LatticeNode[T]] = field(default_factory=list)
    children: list[LatticeNode[T]] = field(default_factory=list)

    def __hash__(self) -> int:
        return hash(id(self))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, LatticeNode):
            return self.element == other.element
        return False

    def __repr__(self) -> str:
        return f"LatticeNode({self.element}, level={self.level})"

    def ancestors(self) -> set[LatticeNode[T]]:
        """Get all ancestors (transitive closure of parents)."""
        result: set[LatticeNode[T]] = set()
        stack = list(self.parents)

        while stack:
            node = stack.pop()
            if node not in result:
                result.add(node)
                stack.extend(node.parents)

        return result

    def descendants(self) -> set[LatticeNode[T]]:
        """Get all descendants (transitive closure of children)."""
        result: set[LatticeNode[T]] = set()
        stack = list(self.children)

        while stack:
            node = stack.pop()
            if node not in result:
                result.add(node)
                stack.extend(node.children)

        return result

    def is_ancestor_of(self, other: LatticeNode[T]) -> bool:
        """Check if this node is an ancestor of other."""
        return self in other.ancestors()

    def is_descendant_of(self, other: LatticeNode[T]) -> bool:
        """Check if this node is a descendant of other."""
        return self in other.descendants()


class ResistanceLattice:
    """Lattice structure for drug resistance hierarchy.

    Models the partial order of resistance profiles based on
    mutation set inclusion.

    Example:
        >>> lattice = ResistanceLattice()
        >>> lattice.add_profile("S1", ["rpoB_S450L"])  # RIF mono
        >>> lattice.add_profile("S2", ["katG_S315T"])  # INH mono
        >>> lattice.add_profile("S3", ["rpoB_S450L", "katG_S315T"])  # MDR
        >>> # S3 is above S1 and S2 in the lattice
    """

    def __init__(self):
        """Initialize empty lattice."""
        self.nodes: dict[frozenset[Mutation], LatticeNode[MutationSet]] = {}
        self.bottom: LatticeNode[MutationSet] | None = None
        self.top: LatticeNode[MutationSet] | None = None

        # Add bottom element (empty set)
        self._add_bottom()

    def _add_bottom(self):
        """Add bottom element (∅, susceptible)."""
        empty = MutationSet.empty()
        self.bottom = LatticeNode(empty, level=0)
        self.nodes[frozenset()] = self.bottom

    def add_profile(
        self,
        name: str,
        mutations: list[str],
    ) -> LatticeNode[MutationSet]:
        """Add a resistance profile to the lattice.

        Args:
            name: Profile identifier
            mutations: List of mutation notation strings

        Returns:
            Node for this profile
        """
        mutation_set = MutationSet.from_strings(mutations, name)
        return self._add_mutation_set(mutation_set)

    def _add_mutation_set(
        self,
        mutation_set: MutationSet,
    ) -> LatticeNode[MutationSet]:
        """Add mutation set to lattice.

        Maintains lattice structure by finding proper position.

        Args:
            mutation_set: Set to add

        Returns:
            Node for this set
        """
        key = mutation_set.to_frozenset()

        # Check if already exists
        if key in self.nodes:
            return self.nodes[key]

        # Create new node
        node = LatticeNode(mutation_set)

        # Find covering elements (immediate parents)
        # These are maximal elements that are subsets of mutation_set
        for _existing_key, existing_node in self.nodes.items():
            existing_set = existing_node.element

            if existing_set.issubset(mutation_set) and existing_set != mutation_set:
                # Check if this is a covering (no intermediate element)
                is_cover = True
                for _other_key, other_node in self.nodes.items():
                    other_set = other_node.element
                    if (
                        existing_set.issubset(other_set)
                        and other_set.issubset(mutation_set)
                        and other_set != existing_set
                        and other_set != mutation_set
                    ):
                        is_cover = False
                        break

                if is_cover:
                    node.parents.append(existing_node)
                    existing_node.children.append(node)

        # Find covered elements (immediate children)
        for _existing_key, existing_node in self.nodes.items():
            existing_set = existing_node.element

            if mutation_set.issubset(existing_set) and existing_set != mutation_set:
                is_cover = True
                for _other_key, other_node in self.nodes.items():
                    other_set = other_node.element
                    if (
                        mutation_set.issubset(other_set)
                        and other_set.issubset(existing_set)
                        and other_set != mutation_set
                        and other_set != existing_set
                    ):
                        is_cover = False
                        break

                if is_cover:
                    node.children.append(existing_node)
                    existing_node.parents.append(node)

        # Compute level
        if node.parents:
            node.level = max(p.level for p in node.parents) + 1
        else:
            node.level = 0

        self.nodes[key] = node

        # Update top if needed
        if self.top is None or node.level > self.top.level:
            self.top = node

        return node

    def meet(
        self,
        a: MutationSet,
        b: MutationSet,
    ) -> MutationSet:
        """Compute meet (greatest lower bound): a ∧ b = a ∩ b

        Args:
            a: First mutation set
            b: Second mutation set

        Returns:
            Intersection (greatest lower bound)
        """
        return a & b

    def join(
        self,
        a: MutationSet,
        b: MutationSet,
    ) -> MutationSet:
        """Compute join (least upper bound): a ∨ b = a ∪ b

        Args:
            a: First mutation set
            b: Second mutation set

        Returns:
            Union (least upper bound)
        """
        return a | b

    def compare(
        self,
        a: MutationSet,
        b: MutationSet,
    ) -> int | None:
        """Compare two elements in lattice ordering.

        Args:
            a: First set
            b: Second set

        Returns:
            -1 if a < b, 0 if a = b, 1 if a > b, None if incomparable
        """
        if a == b:
            return 0
        elif a.issubset(b):
            return -1
        elif b.issubset(a):
            return 1
        else:
            return None  # Incomparable

    def resistance_level(self, mutations: MutationSet) -> ResistanceLevel:
        """Determine resistance level from mutations.

        Args:
            mutations: Mutation set to classify

        Returns:
            Resistance level
        """
        genes = mutations.genes()

        # Check for XDR markers
        has_inh = "katG" in genes or "inhA" in genes
        has_rif = "rpoB" in genes
        has_fq = "gyrA" in genes or "gyrB" in genes
        has_injectable = any(g in genes for g in ["rrs", "eis", "tlyA"])

        if has_inh and has_rif and has_fq and has_injectable:
            return ResistanceLevel.XDR
        elif has_inh and has_rif and (has_fq or has_injectable):
            return ResistanceLevel.PRE_XDR
        elif has_inh and has_rif:
            return ResistanceLevel.MDR
        elif len(genes) > 1:
            return ResistanceLevel.POLY_RESISTANT
        elif len(genes) == 1:
            return ResistanceLevel.MONO_RESISTANT
        else:
            return ResistanceLevel.SUSCEPTIBLE

    def filter_by_level(
        self,
        min_level: int = 0,
        max_level: int | None = None,
    ) -> list[LatticeNode[MutationSet]]:
        """Get nodes within level range.

        Args:
            min_level: Minimum level (inclusive)
            max_level: Maximum level (inclusive, None = no limit)

        Returns:
            List of nodes in range
        """
        result = []
        for node in self.nodes.values():
            if node.level >= min_level and (max_level is None or node.level <= max_level):
                result.append(node)
        return result

    def atoms(self) -> list[LatticeNode[MutationSet]]:
        """Get atoms (elements covering bottom).

        These are single-mutation profiles.

        Returns:
            List of atomic nodes
        """
        if self.bottom is None:
            return []
        return self.bottom.children

    def coatoms(self) -> list[LatticeNode[MutationSet]]:
        """Get coatoms (elements covered by top).

        These are maximal non-full profiles.

        Returns:
            List of coatomic nodes
        """
        if self.top is None:
            return []
        return self.top.parents

    def chains(self) -> list[list[LatticeNode[MutationSet]]]:
        """Find all maximal chains in lattice.

        A chain is a totally ordered subset.

        Returns:
            List of chains (each a list of nodes)
        """
        if self.bottom is None:
            return []

        chains: list[list[LatticeNode[MutationSet]]] = []

        def find_chains(
            current: LatticeNode[MutationSet],
            chain: list[LatticeNode[MutationSet]],
        ):
            chain.append(current)

            if not current.children:
                # End of chain
                chains.append(list(chain))
            else:
                for child in current.children:
                    find_chains(child, chain)

            chain.pop()

        find_chains(self.bottom, [])
        return chains

    def antichains(self) -> list[list[LatticeNode[MutationSet]]]:
        """Find all maximal antichains.

        An antichain is a set of pairwise incomparable elements.

        Returns:
            List of antichains
        """
        # Group by level (elements at same level are candidates for antichain)
        by_level: dict[int, list[LatticeNode[MutationSet]]] = {}
        for node in self.nodes.values():
            if node.level not in by_level:
                by_level[node.level] = []
            by_level[node.level].append(node)

        # Each level is an antichain (in Boolean lattice)
        return list(by_level.values())

    def width(self) -> int:
        """Compute width (size of largest antichain).

        By Dilworth's theorem, this equals the minimum number
        of chains needed to cover the lattice.

        Returns:
            Lattice width
        """
        antichains = self.antichains()
        if not antichains:
            return 0
        return max(len(ac) for ac in antichains)

    def height(self) -> int:
        """Compute height (length of longest chain).

        Returns:
            Lattice height
        """
        if self.top is None:
            return 0
        return self.top.level + 1

    def to_dict(self) -> dict:
        """Convert to dictionary representation.

        Returns:
            Dictionary with lattice structure
        """
        return {
            "nodes": [
                {
                    "mutations": node.element.to_list(),
                    "level": node.level,
                    "parents": [p.element.to_list() for p in node.parents],
                    "children": [c.element.to_list() for c in node.children],
                }
                for node in self.nodes.values()
            ],
            "height": self.height(),
            "width": self.width(),
        }


class PowerSetLattice(Generic[T]):
    """Power set lattice 2^S for any finite set S.

    The power set of S forms a Boolean lattice under subset ordering
    with meet = intersection and join = union.
    """

    def __init__(self, base_set: set[T]):
        """Initialize power set lattice.

        Args:
            base_set: Base set S
        """
        self.base_set = frozenset(base_set)
        self._build_lattice()

    def _build_lattice(self):
        """Build the full power set lattice."""
        self.nodes: dict[frozenset[T], LatticeNode[frozenset[T]]] = {}

        # Generate power set
        elements = list(self.base_set)
        n = len(elements)

        for i in range(2**n):
            subset = frozenset(elements[j] for j in range(n) if (i >> j) & 1)
            node = LatticeNode(subset, level=len(subset))
            self.nodes[subset] = node

        # Add edges (covering relations)
        for subset, node in self.nodes.items():
            for elem in self.base_set - subset:
                larger = subset | {elem}
                if larger in self.nodes:
                    larger_node = self.nodes[larger]
                    node.children.append(larger_node)
                    larger_node.parents.append(node)

        # Set bottom and top
        self.bottom = self.nodes[frozenset()]
        self.top = self.nodes[self.base_set]

    def get_node(self, subset: set[T]) -> LatticeNode[frozenset[T]] | None:
        """Get node for a subset.

        Args:
            subset: Target subset

        Returns:
            Node or None if not valid subset
        """
        return self.nodes.get(frozenset(subset))

    def complement(self, subset: set[T]) -> frozenset[T]:
        """Compute complement.

        Args:
            subset: Subset to complement

        Returns:
            S \\ subset
        """
        return self.base_set - frozenset(subset)

    def is_boolean_sublattice(self, elements: set[frozenset[T]]) -> bool:
        """Check if elements form a Boolean sublattice.

        Args:
            elements: Set of power set elements

        Returns:
            True if closed under meet and join
        """
        for a in elements:
            for b in elements:
                if (a | b) not in elements or (a & b) not in elements:
                    return False
        return True
