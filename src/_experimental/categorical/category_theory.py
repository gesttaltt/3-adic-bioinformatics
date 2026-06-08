# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Category Theory module for compositional neural networks.

This module implements categorical abstractions for designing and
reasoning about neural network architectures compositionally.

Key concepts:
- Objects = Types (tensor shapes, feature spaces)
- Morphisms = Neural network layers/functions
- Functors = Structure-preserving maps between architectures
- Natural transformations = Layer-to-layer mappings

Benefits:
- Type-safe neural network composition
- Formal verification of architecture properties
- Modular, reusable components

References:
- Fong et al. (2019): Backprop as Functor
- Cruttwell et al. (2022): Categorical Foundations of Gradient-Based Learning
- Shiebler et al. (2021): Category Theory in Machine Learning
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F

# Type variables for categorical typing
A = TypeVar("A")  # Object type
B = TypeVar("B")
C = TypeVar("C")
D = TypeVar("D")


@dataclass(frozen=True)
class TensorType:
    """Type representing tensor shape and dtype.

    Acts as an object in the category of tensor spaces.
    """

    shape: tuple[int, ...]
    dtype: torch.dtype = torch.float32
    device: str = "cpu"

    def __repr__(self) -> str:
        return f"TensorType({self.shape}, {self.dtype})"

    def is_compatible(self, tensor: torch.Tensor) -> bool:
        """Check if tensor matches this type (ignoring batch dim)."""
        if tensor.dtype != self.dtype:
            return False
        if len(tensor.shape) < len(self.shape):
            return False
        # Check all but batch dimension
        return tensor.shape[-len(self.shape) :] == self.shape

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> TensorType:
        """Create type from tensor (excluding batch dim)."""
        return cls(
            shape=tuple(tensor.shape[1:]),
            dtype=tensor.dtype,
            device=str(tensor.device),
        )


@dataclass
class Morphism(Generic[A, B]):
    """A morphism (arrow) in a category.

    Represents a function/transformation from source to target.
    In neural networks: layers, activation functions, etc.
    """

    source: A
    target: B
    name: str = ""
    transform: Callable[[torch.Tensor], torch.Tensor] | None = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the morphism."""
        if self.transform is None:
            raise ValueError("No transform defined for this morphism")
        return self.transform(x)

    def compose(self, other: Morphism[B, C]) -> Morphism[A, C]:
        """Compose with another morphism: self ; other = other ∘ self."""
        if self.target != other.source:
            raise TypeError(f"Cannot compose: target {self.target} != source {other.source}")

        def composed_transform(x: torch.Tensor) -> torch.Tensor:
            return other(self(x))

        return Morphism(
            source=self.source,
            target=other.target,
            name=f"({self.name} ; {other.name})",
            transform=composed_transform,
        )


class CategoricalLayer(nn.Module):
    """Neural network layer with categorical typing.

    Enforces type safety: input/output shapes must match declared types.
    """

    def __init__(
        self,
        input_type: TensorType,
        output_type: TensorType,
        layer: nn.Module | None = None,
        name: str = "",
    ):
        """Initialize categorical layer.

        Args:
            input_type: Expected input tensor type
            output_type: Expected output tensor type
            layer: Underlying neural network layer
            name: Layer name for debugging
        """
        super().__init__()
        self.input_type = input_type
        self.output_type = output_type
        self.name = name

        if layer is not None:
            self.layer = layer
        else:
            # Create default linear layer
            in_features = input_type.shape[-1] if input_type.shape else 1
            out_features = output_type.shape[-1] if output_type.shape else 1
            self.layer = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with type checking."""
        # Type check input (optional, can disable for speed)
        if not self.input_type.is_compatible(x):
            raise TypeError(f"Input shape {x.shape} incompatible with {self.input_type}")

        output = self.layer(x)

        # Type check output
        if not self.output_type.is_compatible(output):
            raise TypeError(f"Output shape {output.shape} incompatible with {self.output_type}")

        return output

    def as_morphism(self) -> Morphism[TensorType, TensorType]:
        """Convert to a morphism."""
        return Morphism(
            source=self.input_type,
            target=self.output_type,
            name=self.name,
            transform=self.forward,
        )

    def compose(self, other: CategoricalLayer) -> CategoricalLayer:
        """Compose with another categorical layer."""
        if self.output_type != other.input_type:
            raise TypeError(f"Cannot compose: output {self.output_type} != input {other.input_type}")

        # Create sequential composition
        composed = nn.Sequential(self.layer, other.layer)

        return CategoricalLayer(
            input_type=self.input_type,
            output_type=other.output_type,
            layer=composed,
            name=f"({self.name} ; {other.name})",
        )

    def __rshift__(self, other: CategoricalLayer) -> CategoricalLayer:
        """Compose using >> operator: f >> g = g ∘ f."""
        return self.compose(other)


class Functor(Generic[A, B]):
    """A functor between categories.

    Maps objects to objects and morphisms to morphisms,
    preserving identity and composition.

    In ML context: Maps between network architectures.
    """

    def __init__(
        self,
        object_map: Callable[[A], B],
        morphism_map: Callable[[Morphism], Morphism],
        name: str = "",
    ):
        """Initialize functor.

        Args:
            object_map: How to transform objects
            morphism_map: How to transform morphisms
            name: Functor name
        """
        self.object_map = object_map
        self.morphism_map = morphism_map
        self.name = name

    def apply_object(self, obj: A) -> B:
        """Apply functor to an object."""
        return self.object_map(obj)

    def apply_morphism(self, morph: Morphism) -> Morphism:
        """Apply functor to a morphism."""
        return self.morphism_map(morph)

    def compose(self, other: Functor[B, C]) -> Functor[A, C]:
        """Compose with another functor."""

        def composed_object_map(obj: A) -> C:
            return other.object_map(self.object_map(obj))

        def composed_morphism_map(morph: Morphism) -> Morphism:
            return other.morphism_map(self.morphism_map(morph))

        return Functor(
            object_map=composed_object_map,
            morphism_map=composed_morphism_map,
            name=f"({self.name} ; {other.name})",
        )


class NaturalTransformation(Generic[A, B]):
    """Natural transformation between functors.

    A family of morphisms that commute with functor application.
    In ML: Adapter layers between architectures.
    """

    def __init__(
        self,
        source_functor: Functor,
        target_functor: Functor,
        components: dict[A, Morphism],
        name: str = "",
    ):
        """Initialize natural transformation.

        Args:
            source_functor: Source functor F
            target_functor: Target functor G
            components: For each object A, a morphism F(A) -> G(A)
            name: Transformation name
        """
        self.source_functor = source_functor
        self.target_functor = target_functor
        self.components = components
        self.name = name

    def component(self, obj: A) -> Morphism:
        """Get component at object."""
        return self.components[obj]

    def verify_naturality(self, morphism: Morphism, test_input: torch.Tensor) -> bool:
        """Verify naturality square commutes.

        For f: A -> B, check that:
        η_B ∘ F(f) = G(f) ∘ η_A
        """
        A, B = morphism.source, morphism.target

        # F(f) ; η_B
        Ff = self.source_functor.apply_morphism(morphism)
        eta_B = self.components[B]
        path1 = Ff.compose(eta_B)

        # η_A ; G(f)
        eta_A = self.components[A]
        Gf = self.target_functor.apply_morphism(morphism)
        path2 = eta_A.compose(Gf)

        # Check equality on test input
        result1 = path1(test_input)
        result2 = path2(test_input)

        return torch.allclose(result1, result2, rtol=1e-5)


class ParametricLens(nn.Module):
    """Parametric lens for gradient-based learning.

    Implements the Para construction from categorical learning theory.
    A parametric lens (P, f, f#) consists of:
    - P: Parameter space
    - f: P × A -> B (forward pass)
    - f#: P × A × B* -> P* × A* (backward pass)

    This captures backpropagation categorically.
    """

    def __init__(
        self,
        input_type: TensorType,
        output_type: TensorType,
        param_type: TensorType,
    ):
        """Initialize parametric lens.

        Args:
            input_type: Input tensor type (A)
            output_type: Output tensor type (B)
            param_type: Parameter tensor type (P)
        """
        super().__init__()
        self.input_type = input_type
        self.output_type = output_type
        self.param_type = param_type

        # Parameters
        param_shape = param_type.shape
        self.params = nn.Parameter(torch.randn(*param_shape) * 0.01)

    @abstractmethod
    def forward_pass(self, params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: P × A -> B."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward using stored parameters."""
        return self.forward_pass(self.params, x)


class LinearLens(ParametricLens):
    """Linear layer as a parametric lens."""

    def __init__(self, in_features: int, out_features: int):
        """Initialize linear lens.

        Args:
            in_features: Input dimension
            out_features: Output dimension
        """
        input_type = TensorType((in_features,))
        output_type = TensorType((out_features,))
        param_type = TensorType((out_features, in_features + 1))  # W and b

        super().__init__(input_type, output_type, param_type)

        # Reshape parameters to match expected shapes
        self.params = nn.Parameter(torch.randn(out_features, in_features + 1) * 0.01)

    def forward_pass(self, params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: y = Wx + b."""
        W = params[:, :-1]
        b = params[:, -1]
        return F.linear(x, W, b)


class ProductCategory:
    """Product of two categories.

    Objects are pairs (A, B) and morphisms are pairs (f, g).
    Used for parallel composition of networks.
    """

    @staticmethod
    def product_type(type1: TensorType, type2: TensorType) -> TensorType:
        """Create product type by concatenating shapes."""
        # Concatenate last dimensions
        new_shape = type1.shape[:-1] + (type1.shape[-1] + type2.shape[-1],)
        return TensorType(new_shape, dtype=type1.dtype)

    @staticmethod
    def product_morphism(morph1: Morphism, morph2: Morphism) -> Morphism:
        """Create product morphism (f × g)."""
        source = (morph1.source, morph2.source)
        target = (morph1.target, morph2.target)

        def product_transform(x: torch.Tensor) -> torch.Tensor:
            # Split input
            split_point = morph1.source.shape[-1]
            x1, x2 = x[..., :split_point], x[..., split_point:]
            # Apply morphisms
            y1, y2 = morph1(x1), morph2(x2)
            # Concatenate outputs
            return torch.cat([y1, y2], dim=-1)

        return Morphism(
            source=source,
            target=target,
            name=f"({morph1.name} × {morph2.name})",
            transform=product_transform,
        )


class MonoidalCategory:
    """Monoidal structure on neural network layers.

    Provides tensor product (parallel composition) and unit object.
    """

    @staticmethod
    def tensor_product(layer1: CategoricalLayer, layer2: CategoricalLayer) -> CategoricalLayer:
        """Tensor product of layers (parallel composition).

        (f ⊗ g)(x, y) = (f(x), g(y))
        """
        # Product types
        input_type = ProductCategory.product_type(layer1.input_type, layer2.input_type)
        output_type = ProductCategory.product_type(layer1.output_type, layer2.output_type)

        class ParallelLayer(nn.Module):
            def __init__(self, l1, l2):
                super().__init__()
                self.l1 = l1
                self.l2 = l2
                self.split_point = l1.input_type.shape[-1]

            def forward(self, x):
                x1 = x[..., : self.split_point]
                x2 = x[..., self.split_point :]
                y1 = self.l1(x1)
                y2 = self.l2(x2)
                return torch.cat([y1, y2], dim=-1)

        return CategoricalLayer(
            input_type=input_type,
            output_type=output_type,
            layer=ParallelLayer(layer1, layer2),
            name=f"({layer1.name} ⊗ {layer2.name})",
        )

    @staticmethod
    def unit_object() -> TensorType:
        """Unit object I in the monoidal category."""
        return TensorType(())


class StringDiagram:
    """String diagram representation of neural network.

    String diagrams are a graphical syntax for monoidal categories.
    They make composition and tensor products visual and intuitive.
    """

    def __init__(self):
        """Initialize empty diagram."""
        self.boxes: list[CategoricalLayer] = []
        self.wires: list[tuple[int, int, int, int]] = []
        # (from_box, from_port, to_box, to_port)

    def add_box(self, layer: CategoricalLayer) -> int:
        """Add a box (layer) to the diagram."""
        self.boxes.append(layer)
        return len(self.boxes) - 1

    def connect(self, from_box: int, from_port: int, to_box: int, to_port: int):
        """Connect output port to input port."""
        self.wires.append((from_box, from_port, to_box, to_port))

    def to_sequential(self) -> nn.Sequential:
        """Convert diagram to sequential module.

        Only works for simple sequential diagrams.
        """
        # Topological sort of boxes
        in_degree = {i: 0 for i in range(len(self.boxes))}
        for _, _, to_box, _ in self.wires:
            in_degree[to_box] += 1

        # Find starting boxes
        queue = [i for i, d in in_degree.items() if d == 0]
        order = []

        while queue:
            box = queue.pop(0)
            order.append(box)
            for from_box, _, to_box, _ in self.wires:
                if from_box == box:
                    in_degree[to_box] -= 1
                    if in_degree[to_box] == 0:
                        queue.append(to_box)

        return nn.Sequential(*[self.boxes[i].layer for i in order])

    def visualize(self) -> str:
        """Create ASCII visualization of diagram."""
        lines = []
        for i, box in enumerate(self.boxes):
            lines.append(f"[{i}] {box.name}: {box.input_type} -> {box.output_type}")

        lines.append("\nWires:")
        for from_box, from_port, to_box, to_port in self.wires:
            lines.append(f"  {from_box}:{from_port} -> {to_box}:{to_port}")

        return "\n".join(lines)


class Optic(nn.Module):
    """Optic for bidirectional data flow.

    Generalizes lenses to handle more complex patterns:
    - Prisms for error handling
    - Traversals for batched operations
    - Grates for co-algebraic structures

    In ML: Handles skip connections, attention, residuals.
    """

    def __init__(
        self,
        forward_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        backward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ):
        """Initialize optic.

        Args:
            forward_fn: (x) -> (residual, y) - forward with residual
            backward_fn: (residual, dy) -> dx - backward using residual
        """
        super().__init__()
        self.forward_fn = forward_fn
        self.backward_fn = backward_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (just return output, not residual)."""
        _, y = self.forward_fn(x)
        return y


class ResidualOptic(Optic):
    """Residual connection as an optic.

    Forward: x -> (x, f(x) + x)
    Backward: (x, dy) -> dy + df(x)
    """

    def __init__(self, layer: nn.Module):
        """Initialize residual optic.

        Args:
            layer: Inner layer f
        """
        self.inner_layer = layer

        def forward_fn(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return x, self.inner_layer(x) + x

        def backward_fn(residual: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
            # Gradient flows through both paths
            return dy

        super().__init__(forward_fn, backward_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with residual connection."""
        return self.inner_layer(x) + x


class AttentionOptic(Optic):
    """Self-attention as an optic.

    Models attention as a lens-like structure where:
    - Forward computes attention-weighted values
    - The "residual" is the attention weights (for interpretability)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        """Initialize attention optic.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Projections
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        def forward_fn(
            x: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            batch, seq_len, _ = x.shape

            # Compute QKV
            qkv = self.qkv(x)
            q, k, v = qkv.chunk(3, dim=-1)

            # Reshape for multi-head
            q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

            # Attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)

            # Apply attention
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
            out = self.out_proj(out)

            # Return attention weights as "residual" for interpretability
            return attn, out

        def backward_fn(attn: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
            return dy

        super().__init__(forward_fn, backward_fn)

        # Register submodules
        self._qkv = self.qkv
        self._out_proj = self.out_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning attended values."""
        _, out = self.forward_fn(x)
        return out

    def forward_with_attention(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward returning both output and attention weights."""
        return self.forward_fn(x)


class CategoricalNetwork(nn.Module):
    """Neural network built from categorical components.

    Provides a type-safe way to compose layers with verification.
    """

    def __init__(self, name: str = "CategoricalNet"):
        """Initialize categorical network.

        Args:
            name: Network name
        """
        super().__init__()
        self.name = name
        self.layers: list[CategoricalLayer] = []
        self._modules_list = nn.ModuleList()

    def add_layer(self, layer: CategoricalLayer) -> CategoricalNetwork:
        """Add a layer to the network.

        Args:
            layer: Categorical layer to add

        Returns:
            Self for chaining
        """
        # Type check
        if self.layers:
            last_layer = self.layers[-1]
            if last_layer.output_type != layer.input_type:
                raise TypeError(f"Type mismatch: {last_layer.output_type} != {layer.input_type}")

        self.layers.append(layer)
        self._modules_list.append(layer)
        return self

    @property
    def input_type(self) -> TensorType | None:
        """Get input type of network."""
        return self.layers[0].input_type if self.layers else None

    @property
    def output_type(self) -> TensorType | None:
        """Get output type of network."""
        return self.layers[-1].output_type if self.layers else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer(x)
        return x

    def to_morphism(self) -> Morphism[TensorType, TensorType]:
        """Convert entire network to a morphism."""
        if not self.layers:
            raise ValueError("No layers in network")

        return Morphism(
            source=self.input_type,
            target=self.output_type,
            name=self.name,
            transform=self.forward,
        )

    def verify_types(self) -> bool:
        """Verify all type connections are valid."""
        return all(self.layers[i].output_type == self.layers[i + 1].input_type for i in range(len(self.layers) - 1))
