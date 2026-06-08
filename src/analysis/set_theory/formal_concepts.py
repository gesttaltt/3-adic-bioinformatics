# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Formal Concept Analysis (FCA) for genotype-phenotype relationships.

Formal Concept Analysis provides a mathematical framework for analyzing
binary relations between objects and attributes. In drug resistance:
- Objects = bacterial strains/samples
- Attributes = mutations + resistance phenotypes

A formal concept is a pair (A, B) where:
- A (extent): maximal set of objects sharing attributes B
- B (intent): maximal set of attributes shared by objects A

The concept lattice reveals:
- Mutation-resistance associations
- Hierarchical clustering of resistance profiles
- Implication rules (if mutation X → mutation Y)

References:
    - Ganter & Wille (1999): Formal Concept Analysis
    - Wille (1982): Restructuring Lattice Theory
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FormalContext:
    """Formal context (G, M, I) for concept analysis.

    A formal context consists of:
    - G: Set of objects (genotypes/strains)
    - M: Set of attributes (mutations, resistance phenotypes)
    - I: Binary relation I ⊆ G × M (object has attribute)

    Example:
        >>> ctx = FormalContext()
        >>> ctx.add_object("strain1", {"rpoB_S450L", "katG_S315T", "RIF_R", "INH_R"})
        >>> ctx.add_object("strain2", {"rpoB_S450L", "RIF_R"})
        >>> # Derive shared attributes
        >>> ctx.object_intent({"strain1", "strain2"})
        {"rpoB_S450L", "RIF_R"}
    """

    objects: set[str] = field(default_factory=set)
    attributes: set[str] = field(default_factory=set)
    relation: dict[str, set[str]] = field(default_factory=dict)

    @classmethod
    def from_mutation_data(
        cls,
        samples: dict[str, list[str]],
        resistance: dict[str, list[str]] | None = None,
    ) -> FormalContext:
        """Create context from mutation and resistance data.

        Args:
            samples: Map from sample ID to list of mutations
            resistance: Optional map from sample ID to resistant drugs

        Returns:
            Formal context
        """
        ctx = cls()

        for sample_id, mutations in samples.items():
            attrs: set[str] = set(mutations)

            # Add resistance phenotypes
            if resistance and sample_id in resistance:
                for drug in resistance[sample_id]:
                    attrs.add(f"{drug}_R")

            ctx.add_object(sample_id, attrs)

        return ctx

    @classmethod
    def from_cross_table(
        cls,
        table: list[list[bool]],
        object_names: list[str],
        attribute_names: list[str],
    ) -> FormalContext:
        """Create context from cross table (incidence matrix).

        Args:
            table: Boolean matrix (objects × attributes)
            object_names: Names for rows
            attribute_names: Names for columns

        Returns:
            Formal context
        """
        ctx = cls()

        for i, obj in enumerate(object_names):
            attrs = {attribute_names[j] for j, has_attr in enumerate(table[i]) if has_attr}
            ctx.add_object(obj, attrs)

        return ctx

    def add_object(self, obj: str, attributes: set[str]) -> None:
        """Add object with its attributes.

        Args:
            obj: Object identifier
            attributes: Set of attribute names
        """
        self.objects.add(obj)
        self.attributes.update(attributes)
        self.relation[obj] = attributes.copy()

    def add_attribute(self, attr: str, objects: set[str]) -> None:
        """Add attribute with objects that have it.

        Args:
            attr: Attribute name
            objects: Set of objects with this attribute
        """
        self.attributes.add(attr)
        for obj in objects:
            if obj not in self.relation:
                self.relation[obj] = set()
                self.objects.add(obj)
            self.relation[obj].add(attr)

    def has_attribute(self, obj: str, attr: str) -> bool:
        """Check if object has attribute.

        Args:
            obj: Object identifier
            attr: Attribute name

        Returns:
            True if (obj, attr) ∈ I
        """
        return attr in self.relation.get(obj, set())

    def object_intent(self, objects: set[str]) -> frozenset[str]:
        """Derivation operator for objects: A' = {m ∈ M | ∀g ∈ A: (g,m) ∈ I}

        Get all attributes shared by all objects in A.

        Args:
            objects: Set of objects

        Returns:
            Common attributes (intent)
        """
        if not objects:
            return frozenset(self.attributes)

        result = None
        for obj in objects:
            obj_attrs = self.relation.get(obj, set())
            if result is None:
                result = set(obj_attrs)
            else:
                result &= obj_attrs

        return frozenset(result) if result else frozenset()

    def attribute_extent(self, attributes: set[str]) -> frozenset[str]:
        """Derivation operator for attributes: B' = {g ∈ G | ∀m ∈ B: (g,m) ∈ I}

        Get all objects that have all attributes in B.

        Args:
            attributes: Set of attributes

        Returns:
            Objects with all attributes (extent)
        """
        if not attributes:
            return frozenset(self.objects)

        result = []
        for obj in self.objects:
            obj_attrs = self.relation.get(obj, set())
            if attributes.issubset(obj_attrs):
                result.append(obj)

        return frozenset(result)

    def closure(self, objects: set[str]) -> frozenset[str]:
        """Closure operator: A'' = (A')'

        Args:
            objects: Set of objects

        Returns:
            Closure of objects
        """
        intent = self.object_intent(objects)
        return self.attribute_extent(set(intent))

    def attribute_closure(self, attributes: set[str]) -> frozenset[str]:
        """Attribute closure: B'' = (B')'

        Args:
            attributes: Set of attributes

        Returns:
            Closure of attributes
        """
        extent = self.attribute_extent(attributes)
        return self.object_intent(set(extent))

    def is_closed(self, objects: set[str]) -> bool:
        """Check if object set is closed (A'' = A).

        Args:
            objects: Set of objects

        Returns:
            True if closed
        """
        return self.closure(objects) == frozenset(objects)

    def to_cross_table(self) -> tuple[list[list[bool]], list[str], list[str]]:
        """Convert to cross table representation.

        Returns:
            Tuple of (table, object_names, attribute_names)
        """
        obj_list = sorted(self.objects)
        attr_list = sorted(self.attributes)

        table = [[self.has_attribute(obj, attr) for attr in attr_list] for obj in obj_list]

        return table, obj_list, attr_list

    def subcontext(
        self,
        objects: set[str] | None = None,
        attributes: set[str] | None = None,
    ) -> FormalContext:
        """Extract subcontext.

        Args:
            objects: Subset of objects (None = all)
            attributes: Subset of attributes (None = all)

        Returns:
            Subcontext
        """
        objs = objects if objects is not None else self.objects
        attrs = attributes if attributes is not None else self.attributes

        ctx = FormalContext()
        for obj in objs:
            if obj in self.relation:
                obj_attrs = self.relation[obj] & attrs
                if obj_attrs:
                    ctx.add_object(obj, obj_attrs)

        return ctx


@dataclass(frozen=True)
class FormalConcept:
    """Formal concept (A, B) in a formal context.

    A concept is a pair where:
    - extent A: set of objects
    - intent B: set of attributes
    - A' = B and B' = A (maximality conditions)

    The concepts form a complete lattice under:
    - Subconcept: (A1, B1) ≤ (A2, B2) iff A1 ⊆ A2 (iff B2 ⊆ B1)
    """

    extent: frozenset[str]
    intent: frozenset[str]

    def __repr__(self) -> str:
        ext_str = ", ".join(sorted(self.extent)[:3])
        if len(self.extent) > 3:
            ext_str += f"... ({len(self.extent)} total)"

        int_str = ", ".join(sorted(self.intent)[:3])
        if len(self.intent) > 3:
            int_str += f"... ({len(self.intent)} total)"

        return f"Concept(extent={{{ext_str}}}, intent={{{int_str}}})"

    def is_subconcept_of(self, other: FormalConcept) -> bool:
        """Check if this is a subconcept of other.

        Args:
            other: Potential superconcept

        Returns:
            True if extent ⊆ other.extent
        """
        return self.extent.issubset(other.extent)

    def meet(self, other: FormalConcept, context: FormalContext) -> FormalConcept:
        """Meet (infimum): (A1, B1) ∧ (A2, B2)

        Args:
            other: Other concept
            context: Formal context

        Returns:
            Meet concept
        """
        new_extent = self.extent & other.extent
        new_intent = context.object_intent(set(new_extent))
        return FormalConcept(new_extent, new_intent)

    def join(self, other: FormalConcept, context: FormalContext) -> FormalConcept:
        """Join (supremum): (A1, B1) ∨ (A2, B2)

        Args:
            other: Other concept
            context: Formal context

        Returns:
            Join concept
        """
        new_intent = self.intent & other.intent
        new_extent = context.attribute_extent(set(new_intent))
        return FormalConcept(new_extent, new_intent)

    @property
    def size(self) -> int:
        """Support: number of objects in extent."""
        return len(self.extent)

    @property
    def specificity(self) -> int:
        """Specificity: number of attributes in intent."""
        return len(self.intent)


class ConceptLattice:
    """Lattice of all formal concepts from a context.

    Implements concept mining algorithms to extract all concepts
    and their ordering relationships.
    """

    def __init__(self, context: FormalContext):
        """Initialize concept lattice.

        Args:
            context: Formal context to analyze
        """
        self.context = context
        self.concepts: list[FormalConcept] = []
        self._build_lattice()

    def _build_lattice(self) -> None:
        """Build concept lattice using NextClosure algorithm."""
        # Use simplified Ganter's algorithm
        n = len(self.context.objects)

        # Map objects to indices for lexicographic ordering
        obj_list = sorted(self.context.objects)
        {obj: i for i, obj in enumerate(obj_list)}

        seen_extents: set[frozenset[str]] = set()

        def next_closure(current: frozenset[str]) -> frozenset[str] | None:
            """Find lexicographically next closed set."""
            for i in range(n - 1, -1, -1):
                obj = obj_list[i]

                if obj in current:
                    continue

                # Try adding object i
                candidate = set(current)
                candidate.add(obj)

                # Compute closure
                closure = self.context.closure(candidate)

                # Check if closure is lexicographically valid
                # (no objects before i were added by closure)
                valid = True
                for j in range(i):
                    if obj_list[j] in closure and obj_list[j] not in current:
                        valid = False
                        break

                if valid:
                    return closure

            return None

        # Start with empty set closure (supremum)
        current: frozenset[str] = self.context.closure(set())

        while current is not None:
            if current not in seen_extents:
                intent = self.context.object_intent(set(current))
                concept = FormalConcept(current, intent)
                self.concepts.append(concept)
                seen_extents.add(current)

            current = next_closure(current)

        # Sort by extent size (bottom to top)
        self.concepts.sort(key=lambda c: len(c.extent))

    @property
    def supremum(self) -> FormalConcept | None:
        """Top concept (all objects, common attributes)."""
        if not self.concepts:
            return None
        return max(self.concepts, key=lambda c: len(c.extent))

    @property
    def infimum(self) -> FormalConcept | None:
        """Bottom concept (objects with all attributes, all attributes)."""
        if not self.concepts:
            return None
        return min(self.concepts, key=lambda c: len(c.extent))

    def get_subconcepts(self, concept: FormalConcept) -> list[FormalConcept]:
        """Get immediate subconcepts.

        Args:
            concept: Concept to find subconcepts for

        Returns:
            List of subconcepts
        """
        subconcepts = [c for c in self.concepts if c.extent < concept.extent]

        # Keep only immediate (maximal) subconcepts
        immediate = []
        for c in subconcepts:
            is_immediate = True
            for other in subconcepts:
                if c != other and c.extent < other.extent < concept.extent:
                    is_immediate = False
                    break
            if is_immediate:
                immediate.append(c)

        return immediate

    def get_superconcepts(self, concept: FormalConcept) -> list[FormalConcept]:
        """Get immediate superconcepts.

        Args:
            concept: Concept to find superconcepts for

        Returns:
            List of superconcepts
        """
        superconcepts = [c for c in self.concepts if concept.extent < c.extent]

        # Keep only immediate (minimal) superconcepts
        immediate = []
        for c in superconcepts:
            is_immediate = True
            for other in superconcepts:
                if c != other and concept.extent < other.extent < c.extent:
                    is_immediate = False
                    break
            if is_immediate:
                immediate.append(c)

        return immediate

    def attribute_concepts(self, attribute: str) -> list[FormalConcept]:
        """Get concepts containing an attribute in intent.

        Args:
            attribute: Attribute to search for

        Returns:
            Concepts with this attribute
        """
        return [c for c in self.concepts if attribute in c.intent]

    def object_concepts(self, obj: str) -> list[FormalConcept]:
        """Get concepts containing an object in extent.

        Args:
            obj: Object to search for

        Returns:
            Concepts with this object
        """
        return [c for c in self.concepts if obj in c.extent]

    def introducing_concept(self, attribute: str) -> FormalConcept | None:
        """Get concept introducing an attribute.

        The introducing concept is the smallest concept
        having the attribute in its intent.

        Args:
            attribute: Attribute to find

        Returns:
            Introducing concept or None
        """
        candidates = self.attribute_concepts(attribute)
        if not candidates:
            return None
        return min(candidates, key=lambda c: len(c.extent))

    def generating_concept(self, obj: str) -> FormalConcept | None:
        """Get concept generating an object.

        The generating concept is the largest concept
        having the object in its extent.

        Args:
            obj: Object to find

        Returns:
            Generating concept or None
        """
        candidates = self.object_concepts(obj)
        if not candidates:
            return None
        return max(candidates, key=lambda c: len(c.extent))

    def find_associations(
        self,
        min_support: float = 0.1,
        min_confidence: float = 0.8,
    ) -> list[tuple[set[str], set[str], float, float]]:
        """Find attribute associations (simple association rules).

        Args:
            min_support: Minimum support (fraction of objects)
            min_confidence: Minimum confidence

        Returns:
            List of (antecedent, consequent, support, confidence)
        """
        n_objects = len(self.context.objects)
        associations = []

        for concept in self.concepts:
            if len(concept.intent) < 2:
                continue

            support = len(concept.extent) / n_objects
            if support < min_support:
                continue

            # Try each attribute as consequent
            intent_list = list(concept.intent)
            for i, consequent_attr in enumerate(intent_list):
                antecedent = set(intent_list[:i] + intent_list[i + 1 :])

                if not antecedent:
                    continue

                # Compute confidence
                antecedent_extent = self.context.attribute_extent(antecedent)
                if len(antecedent_extent) == 0:
                    continue

                confidence = len(concept.extent) / len(antecedent_extent)

                if confidence >= min_confidence:
                    associations.append(
                        (
                            antecedent,
                            {consequent_attr},
                            support,
                            confidence,
                        )
                    )

        return associations

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with lattice structure
        """
        return {
            "n_concepts": len(self.concepts),
            "n_objects": len(self.context.objects),
            "n_attributes": len(self.context.attributes),
            "concepts": [
                {
                    "extent": list(c.extent),
                    "intent": list(c.intent),
                    "size": c.size,
                }
                for c in self.concepts
            ],
        }


@dataclass(frozen=True)
class ImplicationRule:
    """Attribute implication: A → B.

    If an object has all attributes in A, it must have all in B.

    This is useful for discovering:
    - Mutation co-occurrence rules
    - Genotype-phenotype implications
    """

    antecedent: frozenset[str]
    consequent: frozenset[str]
    support: float = 0.0
    confidence: float = 1.0

    def __repr__(self) -> str:
        ant = ", ".join(sorted(self.antecedent))
        cons = ", ".join(sorted(self.consequent))
        return f"{{{ant}}} → {{{cons}}} (conf={self.confidence:.2f})"

    def applies_to(self, attributes: set[str]) -> bool:
        """Check if rule applies to attribute set.

        Args:
            attributes: Attributes to check

        Returns:
            True if antecedent is subset of attributes
        """
        return self.antecedent.issubset(attributes)

    def is_satisfied_by(self, attributes: set[str]) -> bool:
        """Check if rule is satisfied by attribute set.

        Args:
            attributes: Attributes to check

        Returns:
            True if applies and consequent holds
        """
        if not self.applies_to(attributes):
            return True  # Vacuously true
        return self.consequent.issubset(attributes)


class ImplicationMiner:
    """Mine implication rules from formal context.

    Implements stem base (Duquenne-Guigues base) computation
    for finding minimal complete set of implications.
    """

    def __init__(self, context: FormalContext):
        """Initialize miner.

        Args:
            context: Formal context to mine
        """
        self.context = context

    def compute_implications(
        self,
        min_support: float = 0.0,
    ) -> list[ImplicationRule]:
        """Compute implication base.

        Args:
            min_support: Minimum support for rules

        Returns:
            List of implication rules
        """
        implications: list[ImplicationRule] = []
        n_objects = len(self.context.objects)

        # For each attribute, find implications
        for attr in self.context.attributes:
            # Objects with this attribute
            extent = self.context.attribute_extent({attr})
            support = len(extent) / n_objects if n_objects > 0 else 0

            if support < min_support:
                continue

            # Common attributes of these objects
            intent = self.context.object_intent(set(extent))

            # The attribute implies all other attributes in intent
            consequent = intent - {attr}
            if consequent:
                rule = ImplicationRule(
                    antecedent=frozenset({attr}),
                    consequent=consequent,
                    support=support,
                    confidence=1.0,  # Exact implication
                )
                implications.append(rule)

        # Find multi-attribute implications
        # (simplified - full stem base is more complex)
        for concept in ConceptLattice(self.context).concepts:
            if len(concept.intent) < 2:
                continue

            support = len(concept.extent) / n_objects if n_objects > 0 else 0
            if support < min_support:
                continue

            # Check for additional implications
            for attr in concept.intent:
                antecedent = concept.intent - {attr}
                closure = self.context.attribute_closure(set(antecedent))

                # If closure adds this attribute, we have implication
                if attr in closure and antecedent:
                    # Check not redundant with existing rules
                    is_new = True
                    for existing in implications:
                        if existing.antecedent.issubset(antecedent) and attr in existing.consequent:
                            is_new = False
                            break

                    if is_new:
                        rule = ImplicationRule(
                            antecedent=antecedent,
                            consequent=frozenset({attr}),
                            support=support,
                            confidence=1.0,
                        )
                        implications.append(rule)

        return implications

    def find_approximate_rules(
        self,
        min_support: float = 0.1,
        min_confidence: float = 0.8,
    ) -> list[ImplicationRule]:
        """Find approximate implication rules.

        Unlike exact implications, these may have confidence < 1.

        Args:
            min_support: Minimum support
            min_confidence: Minimum confidence

        Returns:
            List of approximate rules
        """
        rules: list[ImplicationRule] = []
        n_objects = len(self.context.objects)

        for concept in ConceptLattice(self.context).concepts:
            support = len(concept.extent) / n_objects if n_objects > 0 else 0
            if support < min_support:
                continue

            # Try each attribute as consequent
            for attr in concept.intent:
                antecedent = concept.intent - {attr}
                if not antecedent:
                    continue

                # Compute confidence
                ant_extent = self.context.attribute_extent(set(antecedent))
                if not ant_extent:
                    continue

                confidence = len(concept.extent) / len(ant_extent)

                if confidence >= min_confidence:
                    rule = ImplicationRule(
                        antecedent=antecedent,
                        consequent=frozenset({attr}),
                        support=support,
                        confidence=confidence,
                    )
                    rules.append(rule)

        return rules


class GenotypePhenotypeAnalyzer:
    """Specialized FCA analyzer for genotype-phenotype relationships.

    Analyzes mutation-resistance associations using formal concepts.
    """

    def __init__(
        self,
        samples: dict[str, list[str]],
        resistance: dict[str, list[str]],
    ):
        """Initialize analyzer.

        Args:
            samples: Map sample ID → mutations
            resistance: Map sample ID → resistant drugs
        """
        self.context = FormalContext.from_mutation_data(samples, resistance)
        self.lattice = ConceptLattice(self.context)
        self.miner = ImplicationMiner(self.context)

    def find_resistance_mutations(
        self,
        drug: str,
        min_support: float = 0.1,
    ) -> list[tuple[set[str], float]]:
        """Find mutations associated with resistance to a drug.

        Args:
            drug: Drug name
            min_support: Minimum support

        Returns:
            List of (mutations, support) tuples
        """
        resistance_attr = f"{drug}_R"

        # Find concepts with resistance phenotype
        resistant_concepts = [c for c in self.lattice.concepts if resistance_attr in c.intent]

        results = []
        n_objects = len(self.context.objects)

        for concept in resistant_concepts:
            support = len(concept.extent) / n_objects
            if support < min_support:
                continue

            # Get mutations (exclude resistance phenotypes)
            mutations = {attr for attr in concept.intent if not attr.endswith("_R")}

            if mutations:
                results.append((mutations, support))

        # Sort by support
        results.sort(key=lambda x: -x[1])
        return results

    def find_cross_resistance_rules(
        self,
        min_confidence: float = 0.8,
    ) -> list[ImplicationRule]:
        """Find rules linking resistance to different drugs.

        Args:
            min_confidence: Minimum confidence

        Returns:
            List of cross-resistance implications
        """
        rules = self.miner.find_approximate_rules(min_confidence=min_confidence)

        # Filter to rules about resistance phenotypes
        cross_rules = []
        for rule in rules:
            ant_has_r = any(a.endswith("_R") for a in rule.antecedent)
            cons_has_r = any(c.endswith("_R") for c in rule.consequent)

            if ant_has_r and cons_has_r:
                cross_rules.append(rule)

        return cross_rules

    def mutation_cooccurrence(
        self,
        mutation: str,
        min_support: float = 0.05,
    ) -> list[tuple[str, float]]:
        """Find mutations that co-occur with given mutation.

        Args:
            mutation: Target mutation
            min_support: Minimum support

        Returns:
            List of (co-occurring mutation, support)
        """
        # Objects with this mutation
        extent = self.context.attribute_extent({mutation})
        if not extent:
            return []

        len(extent)
        n_objects = len(self.context.objects)

        # Common attributes
        intent = self.context.object_intent(set(extent))

        results = []
        for attr in intent:
            if attr == mutation or attr.endswith("_R"):
                continue

            # Support is fraction of all objects
            attr_extent = self.context.attribute_extent({attr})
            joint_support = len(extent & attr_extent) / n_objects

            if joint_support >= min_support:
                results.append((attr, joint_support))

        results.sort(key=lambda x: -x[1])
        return results

    def summary(self) -> dict[str, Any]:
        """Generate summary of analysis.

        Returns:
            Summary dictionary
        """
        mutations = [a for a in self.context.attributes if not a.endswith("_R")]
        phenotypes = [a for a in self.context.attributes if a.endswith("_R")]

        return {
            "n_samples": len(self.context.objects),
            "n_mutations": len(mutations),
            "n_phenotypes": len(phenotypes),
            "n_concepts": len(self.lattice.concepts),
            "mutations": mutations[:10],  # First 10
            "phenotypes": phenotypes,
        }
