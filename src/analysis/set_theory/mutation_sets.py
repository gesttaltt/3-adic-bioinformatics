# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Mutation set algebra for resistance analysis.

Provides formal set-theoretic operations on mutations for
understanding drug resistance patterns.

Set operations on mutations are biologically meaningful:
- Union: Combined mutation profile
- Intersection: Shared mutations (potential epistasis)
- Difference: Unique mutations
- Subset: One profile contains another
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum


class MutationType(Enum):
    """Types of genetic mutations."""

    SNP = "snp"  # Single nucleotide polymorphism
    INSERTION = "insertion"  # Insertion
    DELETION = "deletion"  # Deletion
    FRAMESHIFT = "frameshift"  # Frameshift
    PROMOTER = "promoter"  # Promoter region
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Mutation:
    """Immutable representation of a genetic mutation.

    Designed to be hashable for use in sets.

    Attributes:
        gene: Gene name (e.g., "rpoB", "katG")
        position: Position in gene/protein
        reference: Reference allele
        alternate: Alternate allele
        mutation_type: Type of mutation

    Example:
        >>> mut = Mutation("rpoB", 450, "S", "L")
        >>> str(mut)
        'rpoB_S450L'
    """

    gene: str
    position: int
    reference: str = ""
    alternate: str = ""
    mutation_type: MutationType = MutationType.SNP

    def __str__(self) -> str:
        """Standard mutation notation."""
        if self.reference and self.alternate:
            return f"{self.gene}_{self.reference}{self.position}{self.alternate}"
        return f"{self.gene}_{self.position}"

    def __repr__(self) -> str:
        return f"Mutation({str(self)})"

    @classmethod
    def from_string(cls, notation: str) -> Mutation:
        """Parse mutation from standard notation.

        Supports formats:
        - rpoB_S450L (amino acid change)
        - katG_C-15T (nucleotide change, promoter)
        - inhA_315 (position only)

        Args:
            notation: Mutation string

        Returns:
            Mutation object
        """
        # Pattern: gene_RefPosAlt or gene_Pos
        pattern = r"([a-zA-Z0-9]+)_([A-Z]?)(-?\d+)([A-Z]?)"
        match = re.match(pattern, notation)

        if match:
            gene, ref, pos, alt = match.groups()
            position = int(pos)

            # Detect promoter mutations (negative positions)
            mut_type = MutationType.PROMOTER if position < 0 else MutationType.SNP

            return cls(
                gene=gene,
                position=position,
                reference=ref,
                alternate=alt,
                mutation_type=mut_type,
            )

        raise ValueError(f"Cannot parse mutation notation: {notation}")

    def is_in_gene(self, gene: str) -> bool:
        """Check if mutation is in specified gene."""
        return self.gene.lower() == gene.lower()

    def is_synonymous(self) -> bool:
        """Check if mutation is synonymous (same AA)."""
        return self.reference == self.alternate


class MutationSet:
    """Formal set of mutations with algebraic operations.

    Implements set algebra for mutation analysis:
    - Union (|): Combine mutation profiles
    - Intersection (&): Find common mutations
    - Difference (-): Find unique mutations
    - Symmetric difference (^): Mutations in one but not both

    Example:
        >>> profile_a = MutationSet.from_strings(["rpoB_S450L", "katG_S315T"])
        >>> profile_b = MutationSet.from_strings(["rpoB_S450L", "inhA_C-15T"])
        >>> shared = profile_a & profile_b  # {"rpoB_S450L"}
        >>> combined = profile_a | profile_b  # All three mutations
    """

    def __init__(
        self,
        mutations: Iterable[Mutation] | None = None,
        name: str = "",
    ):
        """Initialize mutation set.

        Args:
            mutations: Iterable of Mutation objects
            name: Optional name for this set
        """
        self._mutations: frozenset[Mutation] = frozenset(mutations or [])
        self.name = name

    @classmethod
    def from_strings(
        cls,
        notations: Iterable[str],
        name: str = "",
    ) -> MutationSet:
        """Create from string notations.

        Args:
            notations: Mutation notation strings
            name: Set name

        Returns:
            MutationSet
        """
        mutations = [Mutation.from_string(n) for n in notations]
        return cls(mutations, name)

    @classmethod
    def empty(cls) -> MutationSet:
        """Create empty set (∅)."""
        return cls([], "∅")

    # Set properties

    def __len__(self) -> int:
        return len(self._mutations)

    def __iter__(self):
        return iter(self._mutations)

    def __contains__(self, item: Mutation) -> bool:
        return item in self._mutations

    def __hash__(self) -> int:
        return hash(self._mutations)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, MutationSet):
            return self._mutations == other._mutations
        return False

    def __repr__(self) -> str:
        if self.name:
            return f"MutationSet({self.name}, n={len(self)})"
        return f"MutationSet({{{', '.join(str(m) for m in self._mutations)}}})"

    # Set algebra operations

    def __or__(self, other: MutationSet) -> MutationSet:
        """Union: A ∪ B"""
        return MutationSet(
            self._mutations | other._mutations,
            f"({self.name}∪{other.name})" if self.name and other.name else "",
        )

    def __and__(self, other: MutationSet) -> MutationSet:
        """Intersection: A ∩ B"""
        return MutationSet(
            self._mutations & other._mutations,
            f"({self.name}∩{other.name})" if self.name and other.name else "",
        )

    def __sub__(self, other: MutationSet) -> MutationSet:
        """Difference: A \\ B"""
        return MutationSet(
            self._mutations - other._mutations,
            f"({self.name}\\{other.name})" if self.name and other.name else "",
        )

    def __xor__(self, other: MutationSet) -> MutationSet:
        """Symmetric difference: A △ B"""
        return MutationSet(
            self._mutations ^ other._mutations,
            f"({self.name}△{other.name})" if self.name and other.name else "",
        )

    def issubset(self, other: MutationSet) -> bool:
        """Check if this is a subset: A ⊆ B"""
        return self._mutations.issubset(other._mutations)

    def issuperset(self, other: MutationSet) -> bool:
        """Check if this is a superset: A ⊇ B"""
        return self._mutations.issuperset(other._mutations)

    def isdisjoint(self, other: MutationSet) -> bool:
        """Check if sets are disjoint: A ∩ B = ∅"""
        return self._mutations.isdisjoint(other._mutations)

    # Derived operations

    def union(self, *others: MutationSet) -> MutationSet:
        """Union with multiple sets."""
        result = self._mutations
        for other in others:
            result = result | other._mutations
        return MutationSet(result)

    def intersection(self, *others: MutationSet) -> MutationSet:
        """Intersection with multiple sets."""
        result = self._mutations
        for other in others:
            result = result & other._mutations
        return MutationSet(result)

    def power_set(self) -> list[MutationSet]:
        """Generate power set (all subsets).

        Warning: Exponential in size!

        Returns:
            List of all subsets
        """
        mutations = list(self._mutations)
        n = len(mutations)

        subsets = []
        for i in range(2**n):
            subset = [mutations[j] for j in range(n) if (i >> j) & 1]
            subsets.append(MutationSet(subset))

        return subsets

    # Analysis methods

    def by_gene(self) -> dict[str, MutationSet]:
        """Group mutations by gene.

        Returns:
            Dictionary of gene -> MutationSet
        """
        genes: dict[str, list[Mutation]] = {}
        for mut in self._mutations:
            if mut.gene not in genes:
                genes[mut.gene] = []
            genes[mut.gene].append(mut)

        return {gene: MutationSet(muts, gene) for gene, muts in genes.items()}

    def genes(self) -> set[str]:
        """Get all genes with mutations."""
        return {mut.gene for mut in self._mutations}

    def jaccard_similarity(self, other: MutationSet) -> float:
        """Jaccard similarity: |A ∩ B| / |A ∪ B|

        Args:
            other: Another mutation set

        Returns:
            Similarity in [0, 1]
        """
        intersection = len(self & other)
        union = len(self | other)

        if union == 0:
            return 1.0  # Both empty
        return intersection / union

    def dice_similarity(self, other: MutationSet) -> float:
        """Dice/Sørensen coefficient: 2|A ∩ B| / (|A| + |B|)

        Args:
            other: Another mutation set

        Returns:
            Similarity in [0, 1]
        """
        intersection = len(self & other)
        total = len(self) + len(other)

        if total == 0:
            return 1.0
        return 2 * intersection / total

    def to_list(self) -> list[str]:
        """Convert to list of mutation strings."""
        return [str(m) for m in self._mutations]

    def to_frozenset(self) -> frozenset[Mutation]:
        """Get underlying frozenset."""
        return self._mutations


@dataclass
class ResistanceProfile:
    """Complete resistance profile with mutation sets per drug.

    Maps drugs to their associated mutation sets and
    provides cross-resistance analysis.

    Example:
        >>> profile = ResistanceProfile()
        >>> profile.add_drug("INH", ["katG_S315T", "inhA_C-15T"])
        >>> profile.add_drug("RIF", ["rpoB_S450L"])
        >>> profile.is_mdr()  # True if INH and RIF resistant
    """

    drug_mutations: dict[str, MutationSet] = field(default_factory=dict)
    sample_id: str = ""

    def add_drug(
        self,
        drug: str,
        mutations: MutationSet | list[str],
    ):
        """Add mutations for a drug.

        Args:
            drug: Drug name
            mutations: MutationSet or list of notation strings
        """
        if isinstance(mutations, list):
            mutations = MutationSet.from_strings(mutations, drug)
        self.drug_mutations[drug] = mutations

    def get_mutations(self, drug: str) -> MutationSet:
        """Get mutations for a drug."""
        return self.drug_mutations.get(drug, MutationSet.empty())

    def all_mutations(self) -> MutationSet:
        """Get union of all mutations."""
        if not self.drug_mutations:
            return MutationSet.empty()

        result = MutationSet.empty()
        for mutations in self.drug_mutations.values():
            result = result | mutations
        return result

    def shared_mutations(self, drugs: list[str] | None = None) -> MutationSet:
        """Get mutations shared across drugs.

        Args:
            drugs: Specific drugs to check (None = all)

        Returns:
            Intersection of mutation sets
        """
        if drugs is None:
            drugs = list(self.drug_mutations.keys())

        if not drugs:
            return MutationSet.empty()

        result = self.drug_mutations.get(drugs[0], MutationSet.empty())
        for drug in drugs[1:]:
            result = result & self.drug_mutations.get(drug, MutationSet.empty())

        return result

    def is_resistant(self, drug: str) -> bool:
        """Check if resistant to drug (has mutations)."""
        return len(self.get_mutations(drug)) > 0

    def is_mdr(self) -> bool:
        """Check for Multi-Drug Resistance (INH + RIF)."""
        return self.is_resistant("INH") and self.is_resistant("RIF")

    def is_xdr(self) -> bool:
        """Check for Extensively Drug Resistant.

        MDR + fluoroquinolone + injectable
        """
        if not self.is_mdr():
            return False

        fq_resistant = any(
            self.is_resistant(drug) for drug in ["FQ", "Fluoroquinolone", "Moxifloxacin", "Levofloxacin"]
        )

        injectable_resistant = any(self.is_resistant(drug) for drug in ["Amikacin", "Kanamycin", "Capreomycin"])

        return fq_resistant and injectable_resistant

    def resistance_level(self) -> str:
        """Determine resistance level."""
        if self.is_xdr():
            return "XDR"
        elif self.is_mdr():
            return "MDR"
        elif any(len(m) > 0 for m in self.drug_mutations.values()):
            return "Mono-resistant"
        else:
            return "Susceptible"

    def cross_resistance_matrix(self) -> dict[tuple[str, str], float]:
        """Compute pairwise cross-resistance (Jaccard similarity).

        Returns:
            Dictionary of (drug1, drug2) -> similarity
        """
        drugs = list(self.drug_mutations.keys())
        matrix = {}

        for i, drug1 in enumerate(drugs):
            for drug2 in drugs[i:]:
                sim = self.drug_mutations[drug1].jaccard_similarity(self.drug_mutations[drug2])
                matrix[(drug1, drug2)] = sim
                matrix[(drug2, drug1)] = sim

        return matrix


class MutationSetAlgebra:
    """Advanced set-theoretic operations on mutation sets.

    Provides:
    - Minimal set computation
    - Closure operations
    - Set covers
    - Canonical forms
    """

    @staticmethod
    def minimal_sets(
        sets: list[MutationSet],
        predicate: callable,
    ) -> list[MutationSet]:
        """Find minimal sets satisfying a predicate.

        A set is minimal if no proper subset also satisfies the predicate.

        Args:
            sets: Candidate sets
            predicate: Function returning True if set satisfies condition

        Returns:
            List of minimal satisfying sets
        """
        # Filter to satisfying sets
        satisfying = [s for s in sets if predicate(s)]

        # Remove non-minimal
        minimal = []
        for s in satisfying:
            is_minimal = True
            for other in satisfying:
                if other != s and other.issubset(s) and len(other) < len(s):
                    is_minimal = False
                    break
            if is_minimal:
                minimal.append(s)

        return minimal

    @staticmethod
    def set_cover(
        universe: MutationSet,
        candidates: list[MutationSet],
    ) -> list[MutationSet]:
        """Find minimal set cover (greedy approximation).

        Find smallest collection of candidate sets that cover universe.

        Args:
            universe: Set to cover
            candidates: Available covering sets

        Returns:
            List of sets forming the cover
        """
        uncovered = set(universe._mutations)
        cover = []
        remaining = list(candidates)

        while uncovered and remaining:
            # Greedy: pick set covering most uncovered elements
            best = max(
                remaining,
                key=lambda s: len(set(s._mutations) & uncovered),
            )

            covered = set(best._mutations) & uncovered
            if not covered:
                break

            cover.append(best)
            uncovered -= covered
            remaining.remove(best)

        return cover

    @staticmethod
    def closure(
        seed: MutationSet,
        implications: list[tuple[MutationSet, MutationSet]],
    ) -> MutationSet:
        """Compute closure under implications.

        Given implications A → B, compute all mutations
        derivable from seed.

        Args:
            seed: Initial mutation set
            implications: List of (antecedent, consequent) pairs

        Returns:
            Closure of seed under implications
        """
        result = MutationSet(seed._mutations)
        changed = True

        while changed:
            changed = False
            for antecedent, consequent in implications:
                if antecedent.issubset(result) and not consequent.issubset(result):
                    result = result | consequent
                    changed = True

        return result

    @staticmethod
    def canonical_form(mutation_set: MutationSet) -> MutationSet:
        """Convert to canonical form (sorted, normalized).

        Args:
            mutation_set: Input set

        Returns:
            Canonicalized set
        """
        # Sort mutations by gene then position
        sorted_muts = sorted(
            mutation_set._mutations,
            key=lambda m: (m.gene, m.position),
        )
        return MutationSet(sorted_muts, mutation_set.name)


class CrossResistanceAnalyzer:
    """Analyze cross-resistance patterns using set theory.

    Identifies mutation sets associated with resistance
    to multiple drugs.
    """

    def __init__(self):
        self.profiles: list[ResistanceProfile] = []
        self.drug_sets: dict[str, list[MutationSet]] = {}

    def add_profile(self, profile: ResistanceProfile):
        """Add a resistance profile."""
        self.profiles.append(profile)

        for drug, mutations in profile.drug_mutations.items():
            if drug not in self.drug_sets:
                self.drug_sets[drug] = []
            self.drug_sets[drug].append(mutations)

    def common_mutations(self, drug: str) -> MutationSet:
        """Find mutations common to all resistant samples.

        Args:
            drug: Drug name

        Returns:
            Intersection of all mutation sets for drug
        """
        if drug not in self.drug_sets or not self.drug_sets[drug]:
            return MutationSet.empty()

        result = self.drug_sets[drug][0]
        for mutations in self.drug_sets[drug][1:]:
            result = result & mutations

        return result

    def cross_resistance_mutations(
        self,
        drug1: str,
        drug2: str,
    ) -> MutationSet:
        """Find mutations associated with resistance to both drugs.

        Args:
            drug1: First drug
            drug2: Second drug

        Returns:
            Mutations appearing in both drug's resistant samples
        """
        muts1 = self._all_mutations_for_drug(drug1)
        muts2 = self._all_mutations_for_drug(drug2)

        return muts1 & muts2

    def _all_mutations_for_drug(self, drug: str) -> MutationSet:
        """Get union of all mutations for a drug."""
        if drug not in self.drug_sets:
            return MutationSet.empty()

        result = MutationSet.empty()
        for mutations in self.drug_sets[drug]:
            result = result | mutations
        return result

    def minimal_resistance_sets(
        self,
        drug: str,
        all_known_mutations: MutationSet,
    ) -> list[MutationSet]:
        """Find minimal mutation sets conferring resistance.

        Uses set cover to find smallest mutation combinations.

        Args:
            drug: Drug name
            all_known_mutations: Universe of possible mutations

        Returns:
            List of minimal resistance-conferring sets
        """
        if drug not in self.drug_sets:
            return []

        # Generate power set of common mutations
        common = self.common_mutations(drug)

        # Find minimal sets that appear in resistant samples
        def is_resistance_marker(s: MutationSet) -> bool:
            # Check if this set appears in most resistant samples
            count = sum(1 for mutations in self.drug_sets[drug] if s.issubset(mutations))
            return count >= len(self.drug_sets[drug]) * 0.5

        # Only consider subsets up to size 3 for efficiency
        candidates = []
        mutations = list(common._mutations)
        for size in range(1, min(4, len(mutations) + 1)):
            from itertools import combinations

            for combo in combinations(mutations, size):
                candidates.append(MutationSet(combo))

        return MutationSetAlgebra.minimal_sets(candidates, is_resistance_marker)

    def similarity_matrix(self) -> dict[tuple[str, str], float]:
        """Compute drug similarity based on mutation overlap.

        Returns:
            Pairwise Jaccard similarities
        """
        drugs = list(self.drug_sets.keys())
        matrix = {}

        for i, drug1 in enumerate(drugs):
            all_muts1 = self._all_mutations_for_drug(drug1)
            for drug2 in drugs[i:]:
                all_muts2 = self._all_mutations_for_drug(drug2)
                sim = all_muts1.jaccard_similarity(all_muts2)
                matrix[(drug1, drug2)] = sim
                matrix[(drug2, drug1)] = sim

        return matrix
