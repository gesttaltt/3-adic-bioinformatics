# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Categorical Functors for Biological Modeling.

Implements functors (structure-preserving maps between categories)
for biological sequence and structure analysis.

Key Concepts:
1. Category: Collection of objects and morphisms
2. Functor: Map F: C -> D preserving composition and identities
3. Natural Transformation: Map between functors α: F => G

Biological Categories:
- Seq: Category of sequences (objects = sequences, morphisms = alignments)
- Prot: Category of proteins (objects = proteins, morphisms = homology)
- Lat: Category of latent spaces (objects = embeddings, morphisms = continuous maps)

Important Functors:
- Encode: Seq -> Lat (encoder)
- Decode: Lat -> Seq (decoder)
- Fold: Seq -> Struct (structure prediction)
- Translate: DNA -> Protein (genetic code)

The categorical perspective ensures:
- Compositionality: Complex operations from simple ones
- Naturality: Transformations respect structure
- Universality: Canonical constructions (limits, colimits)

References:
- MacLane (1971): Categories for the Working Mathematician
- Spivak (2014): Category Theory for the Sciences
- Fong & Spivak (2019): Seven Sketches in Compositionality
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

import torch
import torch.nn as nn

# Type variables for categorical generics
Obj = TypeVar("Obj")  # Object type
Mor = TypeVar("Mor")  # Morphism type
A = TypeVar("A")
B = TypeVar("B")


@dataclass
class CategoryObject:
    """Object in a category."""

    name: str
    data: any = None
    properties: dict[str, any] = field(default_factory=dict)


@dataclass
class Morphism(Generic[A, B]):
    """Morphism between objects in a category.

    Represents a structure-preserving map from source to target.
    """

    source: A
    target: B
    name: str = ""
    map_fn: Callable[[A], B] | None = None

    def apply(self, x: A) -> B:
        """Apply morphism to object."""
        if self.map_fn is not None:
            return self.map_fn(x)
        raise NotImplementedError("No map function defined")


class Category(ABC):
    """Abstract base class for a category.

    A category C consists of:
    - Objects: ob(C)
    - Morphisms: for each pair A, B in ob(C), a set Hom(A, B)
    - Composition: for f: A->B and g: B->C, g∘f: A->C
    - Identity: for each A, id_A: A->A
    """

    @abstractmethod
    def objects(self) -> list[CategoryObject]:
        """Return all objects in the category."""
        pass

    @abstractmethod
    def morphisms(self, source: CategoryObject, target: CategoryObject) -> list[Morphism]:
        """Return all morphisms between two objects."""
        pass

    @abstractmethod
    def identity(self, obj: CategoryObject) -> Morphism:
        """Return identity morphism for an object."""
        pass

    @abstractmethod
    def compose(self, f: Morphism, g: Morphism) -> Morphism:
        """Compose two morphisms: g ∘ f."""
        pass


class SequenceCategory(Category):
    """Category of biological sequences.

    Objects: Sequences (DNA, RNA, protein)
    Morphisms: Sequence transformations (mutations, alignments)
    """

    def __init__(self, sequence_type: str = "protein"):
        """Initialize sequence category.

        Args:
            sequence_type: Type of sequences ('dna', 'rna', 'protein')
        """
        self.sequence_type = sequence_type
        self._objects: dict[str, CategoryObject] = {}
        self._morphisms: dict[tuple[str, str], list[Morphism]] = {}

        # Define alphabet
        if sequence_type == "protein":
            self.alphabet = set("ACDEFGHIKLMNPQRSTVWY")
        elif sequence_type == "dna":
            self.alphabet = set("ACGT")
        elif sequence_type == "rna":
            self.alphabet = set("ACGU")
        else:
            self.alphabet = set()

    def add_object(self, name: str, sequence: str) -> CategoryObject:
        """Add a sequence object to the category.

        Args:
            name: Object identifier
            sequence: Sequence string

        Returns:
            Created CategoryObject
        """
        obj = CategoryObject(
            name=name,
            data=sequence,
            properties={"length": len(sequence)},
        )
        self._objects[name] = obj
        return obj

    def add_morphism(
        self,
        source: CategoryObject,
        target: CategoryObject,
        name: str,
        transform: Callable[[str], str],
    ) -> Morphism:
        """Add a morphism between sequences.

        Args:
            source: Source sequence object
            target: Target sequence object
            name: Morphism name
            transform: Transformation function

        Returns:
            Created Morphism
        """
        mor = Morphism(
            source=source,
            target=target,
            name=name,
            map_fn=transform,
        )

        key = (source.name, target.name)
        if key not in self._morphisms:
            self._morphisms[key] = []
        self._morphisms[key].append(mor)

        return mor

    def objects(self) -> list[CategoryObject]:
        return list(self._objects.values())

    def morphisms(self, source: CategoryObject, target: CategoryObject) -> list[Morphism]:
        key = (source.name, target.name)
        return self._morphisms.get(key, [])

    def identity(self, obj: CategoryObject) -> Morphism:
        return Morphism(
            source=obj,
            target=obj,
            name=f"id_{obj.name}",
            map_fn=lambda x: x,
        )

    def compose(self, f: Morphism, g: Morphism) -> Morphism:
        """Compose g after f: (g ∘ f)(x) = g(f(x))."""
        if f.target != g.source:
            raise ValueError("Cannot compose: f.target != g.source")

        return Morphism(
            source=f.source,
            target=g.target,
            name=f"{g.name}∘{f.name}",
            map_fn=lambda x: g.apply(f.apply(x)),
        )


class CategoricalFunctor(nn.Module, ABC):
    """Functor between categories implemented as neural network.

    A functor F: C -> D consists of:
    - Object map: For each A in C, an object F(A) in D
    - Morphism map: For each f: A->B in C, a morphism F(f): F(A)->F(B) in D

    Satisfying:
    - F(id_A) = id_{F(A)}
    - F(g ∘ f) = F(g) ∘ F(f)
    """

    def __init__(
        self,
        source_category: Category,
        target_category: Category,
    ):
        """Initialize functor.

        Args:
            source_category: Source category C
            target_category: Target category D
        """
        super().__init__()
        self.source = source_category
        self.target = target_category

    @abstractmethod
    def map_object(self, obj: CategoryObject) -> CategoryObject:
        """Map object from source to target category."""
        pass

    @abstractmethod
    def map_morphism(self, mor: Morphism) -> Morphism:
        """Map morphism from source to target category."""
        pass

    def forward(self, x: any) -> any:
        """Apply functor to input."""
        if isinstance(x, CategoryObject):
            return self.map_object(x)
        elif isinstance(x, Morphism):
            return self.map_morphism(x)
        else:
            raise TypeError(f"Cannot apply functor to {type(x)}")


class NaturalTransformation(nn.Module):
    """Natural transformation between functors.

    For functors F, G: C -> D, a natural transformation α: F => G
    assigns to each object A in C a morphism α_A: F(A) -> G(A) in D
    such that for any morphism f: A -> B in C:

        α_B ∘ F(f) = G(f) ∘ α_A

    (naturality square commutes)
    """

    def __init__(
        self,
        source_functor: CategoricalFunctor,
        target_functor: CategoricalFunctor,
        hidden_dim: int = 64,
    ):
        """Initialize natural transformation.

        Args:
            source_functor: Functor F
            target_functor: Functor G
            hidden_dim: Hidden dimension for component maps
        """
        super().__init__()
        self.source_functor = source_functor
        self.target_functor = target_functor

        # Component maps (learned)
        self.component_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def component(
        self,
        obj: CategoryObject,
    ) -> Morphism:
        """Get component morphism for an object.

        Returns α_A: F(A) -> G(A)

        Args:
            obj: Object in source category

        Returns:
            Component morphism
        """
        f_obj = self.source_functor.map_object(obj)
        g_obj = self.target_functor.map_object(obj)

        def transform(x):
            if isinstance(x, torch.Tensor):
                return self.component_net(x)
            return x

        return Morphism(
            source=f_obj,
            target=g_obj,
            name=f"α_{obj.name}",
            map_fn=transform,
        )

    def check_naturality(
        self,
        f: Morphism,
        tolerance: float = 1e-5,
    ) -> bool:
        """Check if naturality square commutes for a morphism.

        For f: A -> B, checks that:
            α_B ∘ F(f) = G(f) ∘ α_A

        Args:
            f: Morphism to check
            tolerance: Tolerance for equality check

        Returns:
            True if naturality holds
        """
        # Get component morphisms
        alpha_A = self.component(f.source)
        alpha_B = self.component(f.target)

        # Map f through functors
        F_f = self.source_functor.map_morphism(f)
        G_f = self.target_functor.map_morphism(f)

        # Check commutativity on a test input
        test_input = torch.randn(64)

        # Path 1: F(f) then α_B
        path1 = alpha_B.apply(F_f.apply(test_input))

        # Path 2: α_A then G(f)
        path2 = G_f.apply(alpha_A.apply(test_input))

        # Check equality
        if isinstance(path1, torch.Tensor) and isinstance(path2, torch.Tensor):
            return torch.allclose(path1, path2, atol=tolerance)

        return path1 == path2

    def forward(
        self,
        obj: CategoryObject,
        data: torch.Tensor,
    ) -> torch.Tensor:
        """Apply natural transformation component.

        Args:
            obj: Object in source category
            data: Data to transform

        Returns:
            Transformed data
        """
        return self.component(obj).apply(data)


class CodonToProteinFunctor(CategoricalFunctor):
    """Functor from codon sequences to protein sequences.

    Implements the genetic code as a functor:
    - Objects: Codon sequences -> Protein sequences
    - Morphisms: Codon mutations -> Amino acid changes
    """

    # Standard genetic code
    CODON_TABLE = {
        "TTT": "F",
        "TTC": "F",
        "TTA": "L",
        "TTG": "L",
        "CTT": "L",
        "CTC": "L",
        "CTA": "L",
        "CTG": "L",
        "ATT": "I",
        "ATC": "I",
        "ATA": "I",
        "ATG": "M",
        "GTT": "V",
        "GTC": "V",
        "GTA": "V",
        "GTG": "V",
        "TCT": "S",
        "TCC": "S",
        "TCA": "S",
        "TCG": "S",
        "CCT": "P",
        "CCC": "P",
        "CCA": "P",
        "CCG": "P",
        "ACT": "T",
        "ACC": "T",
        "ACA": "T",
        "ACG": "T",
        "GCT": "A",
        "GCC": "A",
        "GCA": "A",
        "GCG": "A",
        "TAT": "Y",
        "TAC": "Y",
        "TAA": "*",
        "TAG": "*",
        "CAT": "H",
        "CAC": "H",
        "CAA": "Q",
        "CAG": "Q",
        "AAT": "N",
        "AAC": "N",
        "AAA": "K",
        "AAG": "K",
        "GAT": "D",
        "GAC": "D",
        "GAA": "E",
        "GAG": "E",
        "TGT": "C",
        "TGC": "C",
        "TGA": "*",
        "TGG": "W",
        "CGT": "R",
        "CGC": "R",
        "CGA": "R",
        "CGG": "R",
        "AGT": "S",
        "AGC": "S",
        "AGA": "R",
        "AGG": "R",
        "GGT": "G",
        "GGC": "G",
        "GGA": "G",
        "GGG": "G",
    }

    def __init__(self):
        """Initialize codon to protein functor."""
        source = SequenceCategory("dna")
        target = SequenceCategory("protein")
        super().__init__(source, target)

    def translate(self, dna_sequence: str) -> str:
        """Translate DNA to protein.

        Args:
            dna_sequence: DNA sequence

        Returns:
            Protein sequence
        """
        protein = []
        for i in range(0, len(dna_sequence) - 2, 3):
            codon = dna_sequence[i : i + 3].upper()
            aa = self.CODON_TABLE.get(codon, "X")
            if aa == "*":  # Stop codon
                break
            protein.append(aa)
        return "".join(protein)

    def map_object(self, obj: CategoryObject) -> CategoryObject:
        """Map DNA sequence to protein sequence.

        Args:
            obj: DNA sequence object

        Returns:
            Protein sequence object
        """
        protein_seq = self.translate(obj.data)
        return CategoryObject(
            name=f"protein_{obj.name}",
            data=protein_seq,
            properties={"length": len(protein_seq), "source": obj.name},
        )

    def map_morphism(self, mor: Morphism) -> Morphism:
        """Map DNA mutation to protein mutation.

        Args:
            mor: DNA morphism (mutation)

        Returns:
            Protein morphism (amino acid change)
        """

        def protein_transform(dna_seq):
            # Apply DNA transformation then translate
            mutated = mor.apply(dna_seq)
            return self.translate(mutated)

        return Morphism(
            source=self.map_object(mor.source),
            target=self.map_object(mor.target),
            name=f"translate({mor.name})",
            map_fn=protein_transform,
        )


class LatentSpaceFunctor(CategoricalFunctor, nn.Module):
    """Functor from sequences to latent space.

    Implements encoder as a functor:
    - Objects: Sequences -> Latent embeddings
    - Morphisms: Sequence transforms -> Continuous maps
    """

    def __init__(
        self,
        vocab_size: int = 21,
        embed_dim: int = 64,
        latent_dim: int = 16,
    ):
        """Initialize latent space functor.

        Args:
            vocab_size: Size of sequence vocabulary
            embed_dim: Embedding dimension
            latent_dim: Latent space dimension
        """
        nn.Module.__init__(self)
        # Categorical structure (documented for mathematical clarity):
        # source: SequenceCategory("protein") - protein sequences
        # target: Latent space category (implicit Euclidean/hyperbolic)

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim

        # Encoder network
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, latent_dim),
        )

        # For morphism mapping
        self.morphism_encoder = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def sequence_to_indices(self, sequence: str) -> torch.Tensor:
        """Convert sequence to tensor of indices.

        Args:
            sequence: Amino acid sequence

        Returns:
            Index tensor
        """
        aa_to_idx = {aa: i + 1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        indices = [aa_to_idx.get(aa, 0) for aa in sequence.upper()]
        return torch.tensor(indices)

    def encode(self, sequence: str) -> torch.Tensor:
        """Encode sequence to latent vector.

        Args:
            sequence: Sequence string

        Returns:
            Latent embedding
        """
        indices = self.sequence_to_indices(sequence)
        embedded = self.embedding(indices)
        pooled = embedded.mean(dim=0)
        return self.encoder(pooled)

    def map_object(self, obj: CategoryObject) -> CategoryObject:
        """Map sequence to latent embedding.

        Args:
            obj: Sequence object

        Returns:
            Latent embedding object
        """
        embedding = self.encode(obj.data)
        return CategoryObject(
            name=f"latent_{obj.name}",
            data=embedding,
            properties={"source": obj.name, "dim": self.latent_dim},
        )

    def map_morphism(self, mor: Morphism) -> Morphism:
        """Map sequence morphism to latent space morphism.

        Args:
            mor: Sequence morphism

        Returns:
            Latent space morphism
        """
        source_latent = self.map_object(mor.source)
        target_latent = self.map_object(mor.target)

        def latent_transform(z):
            # Learn transformation in latent space
            combined = torch.cat([z, target_latent.data], dim=-1)
            return self.morphism_encoder(combined)

        return Morphism(
            source=source_latent,
            target=target_latent,
            name=f"latent({mor.name})",
            map_fn=latent_transform,
        )

    def forward(self, sequence: str) -> torch.Tensor:
        """Forward pass: encode sequence.

        Args:
            sequence: Input sequence

        Returns:
            Latent embedding
        """
        return self.encode(sequence)


__all__ = [
    "CategoryObject",
    "Morphism",
    "Category",
    "SequenceCategory",
    "CategoricalFunctor",
    "NaturalTransformation",
    "CodonToProteinFunctor",
    "LatentSpaceFunctor",
]
