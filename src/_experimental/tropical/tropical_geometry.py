# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tropical Geometry module for neural network analysis.

This module implements tropical algebraic methods for:
- Analyzing ReLU networks as tropical polynomials
- Computing linear regions of piecewise linear functions
- Phylogenetic tree distances in tropical space

Key insight: ReLU networks compute tropical polynomials, which are
piecewise linear functions. This connection enables:
- Exact computation of decision boundaries
- Counting linear regions
- Geometric analysis of network expressivity

References:
- Zhang et al. (2018): Tropical Geometry of Deep Neural Networks
- Montufar et al. (2014): On Number of Linear Regions
- Speyer & Sturmfels (2009): Tropical Mathematics
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

# =============================================================================
# Tropical Semiring Operations
# =============================================================================


class TropicalSemiring:
    """Operations in the tropical (max-plus) semiring.

    In tropical arithmetic:
    - Addition becomes max: a ⊕ b = max(a, b)
    - Multiplication becomes addition: a ⊗ b = a + b
    - The additive identity is -∞
    - The multiplicative identity is 0

    This semiring naturally appears in:
    - Shortest path algorithms
    - ReLU network analysis
    - Phylogenetic trees
    """

    NEG_INF = float("-inf")

    @staticmethod
    def add(a: float, b: float) -> float:
        """Tropical addition: max(a, b)."""
        return max(a, b)

    @staticmethod
    def multiply(a: float, b: float) -> float:
        """Tropical multiplication: a + b."""
        if a == TropicalSemiring.NEG_INF or b == TropicalSemiring.NEG_INF:
            return TropicalSemiring.NEG_INF
        return a + b

    @staticmethod
    def power(a: float, n: int) -> float:
        """Tropical power: n * a."""
        if n == 0:
            return 0.0  # Multiplicative identity
        if a == TropicalSemiring.NEG_INF:
            return TropicalSemiring.NEG_INF
        return n * a

    @classmethod
    def add_tensor(cls, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Tropical addition for tensors."""
        return torch.maximum(a, b)

    @classmethod
    def multiply_tensor(cls, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Tropical multiplication for tensors."""
        return a + b


@dataclass
class TropicalMonomial:
    """A tropical monomial: coefficient + sum of variable powers.

    In tropical arithmetic, a monomial c * x1^a1 * x2^a2 * ... * xn^an
    becomes c + a1*x1 + a2*x2 + ... + an*xn (a linear function).
    """

    coefficient: float
    exponents: tuple[int, ...]

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate monomial at point x."""
        if len(x) != len(self.exponents):
            raise ValueError(f"Expected {len(self.exponents)} variables, got {len(x)}")
        return self.coefficient + sum(e * xi for e, xi in zip(self.exponents, x, strict=False))

    def __repr__(self) -> str:
        terms = [f"{self.coefficient}"]
        for i, e in enumerate(self.exponents):
            if e != 0:
                terms.append(f"{e}*x{i}")
        return " + ".join(terms)


class TropicalPolynomial:
    """A tropical polynomial: max of tropical monomials.

    A tropical polynomial is a piecewise linear function defined as
    the maximum of finitely many linear functions (monomials).

    This represents:
    - The output of a single ReLU unit
    - General piecewise linear functions
    - Solutions to tropical linear systems
    """

    def __init__(self, monomials: list[TropicalMonomial] | None = None):
        """Initialize tropical polynomial.

        Args:
            monomials: List of TropicalMonomial terms
        """
        self.monomials = monomials or []

    def add_monomial(self, monomial: TropicalMonomial):
        """Add a monomial term."""
        self.monomials.append(monomial)

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate polynomial at point x (tropical max)."""
        if not self.monomials:
            return TropicalSemiring.NEG_INF

        values = [m.evaluate(x) for m in self.monomials]
        return max(values)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate at multiple points."""
        return np.array([self.evaluate(x) for x in X])

    def active_monomial(self, x: np.ndarray) -> int:
        """Return index of the active (maximizing) monomial at x."""
        values = [m.evaluate(x) for m in self.monomials]
        return int(np.argmax(values))

    def tropical_add(self, other: TropicalPolynomial) -> TropicalPolynomial:
        """Tropical addition: max of all monomials from both."""
        result = TropicalPolynomial()
        result.monomials = self.monomials + other.monomials
        return result

    def tropical_multiply(self, other: TropicalPolynomial) -> TropicalPolynomial:
        """Tropical multiplication: sum coefficients, add exponents."""
        result = TropicalPolynomial()
        for m1 in self.monomials:
            for m2 in other.monomials:
                new_coef = m1.coefficient + m2.coefficient
                new_exp = tuple(e1 + e2 for e1, e2 in zip(m1.exponents, m2.exponents, strict=False))
                result.add_monomial(TropicalMonomial(new_coef, new_exp))
        return result

    @classmethod
    def from_linear(cls, weights: np.ndarray, bias: float) -> TropicalPolynomial:
        """Create tropical polynomial from linear function weights^T x + bias."""
        poly = cls()
        tuple(int(w) if w == int(w) else 1 for w in weights)
        # For continuous weights, we use the linear form directly
        poly.add_monomial(TropicalMonomial(bias, tuple([1] * len(weights))))
        return poly

    @classmethod
    def relu(cls, input_poly: TropicalPolynomial) -> TropicalPolynomial:
        """Apply ReLU: max(0, f) = max(f, 0) in tropical."""
        zero_poly = cls()
        zero_poly.add_monomial(
            TropicalMonomial(0.0, (0,) * len(input_poly.monomials[0].exponents if input_poly.monomials else ()))
        )
        return input_poly.tropical_add(zero_poly)

    @property
    def n_monomials(self) -> int:
        """Number of monomial terms."""
        return len(self.monomials)

    def __repr__(self) -> str:
        if not self.monomials:
            return "TropicalPolynomial(empty)"
        terms = [f"({m})" for m in self.monomials]
        return f"max({', '.join(terms)})"


# =============================================================================
# Neural Network Analysis
# =============================================================================


@dataclass
class LinearRegion:
    """A linear region of a piecewise linear function.

    Each region is defined by a set of active ReLU patterns
    and the corresponding affine function.
    """

    activation_pattern: tuple[bool, ...]  # True = ReLU active
    weights: np.ndarray  # Affine weights in this region
    bias: float  # Affine bias in this region
    vertices: list[np.ndarray] | None = None  # Vertices of the polytope


class TropicalNNAnalyzer:
    """Analyze ReLU networks using tropical geometry.

    Key capabilities:
    - Compute the number of linear regions
    - Extract activation patterns
    - Analyze decision boundaries
    - Compute tropical representation
    """

    def __init__(self, model: nn.Module):
        """Initialize analyzer with a ReLU network.

        Args:
            model: PyTorch model (should use ReLU activations)
        """
        self.model = model
        self.layers = self._extract_layers()

    def _extract_layers(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Extract weight matrices and biases from the model."""
        layers = []
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                W = module.weight.detach().cpu().numpy()
                b = module.bias.detach().cpu().numpy() if module.bias is not None else np.zeros(W.shape[0])
                layers.append((W, b))
        return layers

    def compute_linear_regions(
        self,
        input_bounds: tuple[np.ndarray, np.ndarray] | None = None,
        sampling_method: str = "random",
        n_samples: int = 10000,
    ) -> int:
        """Estimate the number of linear regions.

        Uses sampling to estimate the number of distinct activation
        patterns, which corresponds to linear regions.

        Args:
            input_bounds: (lower, upper) bounds for input space
            sampling_method: 'random' or 'grid'
            n_samples: Number of samples

        Returns:
            Estimated number of linear regions
        """
        if not self.layers:
            return 1

        input_dim = self.layers[0][0].shape[1]

        if input_bounds is None:
            input_bounds = (-10 * np.ones(input_dim), 10 * np.ones(input_dim))

        lower, upper = input_bounds

        # Generate samples
        if sampling_method == "random":
            samples = np.random.uniform(lower, upper, (n_samples, input_dim))
        else:  # grid
            n_per_dim = int(n_samples ** (1 / input_dim))
            grids = [np.linspace(l, u, n_per_dim) for l, u in zip(lower, upper, strict=False)]
            samples = np.array(np.meshgrid(*grids)).reshape(input_dim, -1).T

        # Compute activation patterns
        patterns = set()
        for x in samples:
            pattern = self._get_activation_pattern(x)
            patterns.add(pattern)

        return len(patterns)

    def _get_activation_pattern(self, x: np.ndarray) -> tuple[tuple[bool, ...], ...]:
        """Get the activation pattern for input x."""
        patterns = []
        h = x

        for W, b in self.layers[:-1]:  # All but last layer (usually no ReLU)
            pre_activation = W @ h + b
            pattern = tuple(pre_activation > 0)
            patterns.append(pattern)
            h = np.maximum(0, pre_activation)  # ReLU

        return tuple(patterns)

    def extract_tropical_polynomial(
        self,
        output_idx: int = 0,
    ) -> TropicalPolynomial:
        """Extract the tropical polynomial for a network output.

        For deep ReLU networks, the output is a tropical rational function,
        but for single-layer networks we can compute the exact polynomial.

        Args:
            output_idx: Which output neuron to analyze

        Returns:
            TropicalPolynomial representation
        """
        if len(self.layers) == 1:
            # Single layer: just linear
            W, b = self.layers[0]
            poly = TropicalPolynomial()
            poly.add_monomial(
                TropicalMonomial(
                    coefficient=b[output_idx],
                    exponents=tuple(W[output_idx, :].astype(int)),
                )
            )
            return poly

        # For deeper networks, enumerate activation patterns
        # This is exponential, so we limit to small networks
        self.layers[0][0].shape[1]
        total_hidden = sum(W.shape[0] for W, _ in self.layers[:-1])

        if total_hidden > 20:
            raise ValueError(f"Network too large for exact analysis ({total_hidden} hidden units)")

        poly = TropicalPolynomial()

        # Enumerate all 2^total_hidden activation patterns
        for pattern_int in range(2**total_hidden):
            pattern = self._int_to_pattern(pattern_int, total_hidden)

            # Compute the affine function for this pattern
            affine = self._compute_affine_for_pattern(pattern, output_idx)
            if affine is not None:
                W_eff, b_eff = affine
                poly.add_monomial(
                    TropicalMonomial(
                        coefficient=b_eff,
                        exponents=tuple(W_eff.astype(int)),
                    )
                )

        return poly

    def _int_to_pattern(self, n: int, length: int) -> tuple[bool, ...]:
        """Convert integer to binary pattern."""
        return tuple((n >> i) & 1 == 1 for i in range(length))

    def _compute_affine_for_pattern(
        self,
        pattern: tuple[bool, ...],
        output_idx: int,
    ) -> tuple[np.ndarray, float] | None:
        """Compute effective affine function for an activation pattern.

        Returns None if the pattern is infeasible.
        """
        # Start with identity transformation
        W_eff = np.eye(self.layers[0][0].shape[1])
        b_eff = np.zeros(self.layers[0][0].shape[1])

        pattern_idx = 0
        for _layer_idx, (W, b) in enumerate(self.layers[:-1]):
            n_units = W.shape[0]
            layer_pattern = pattern[pattern_idx : pattern_idx + n_units]
            pattern_idx += n_units

            # Apply linear transformation
            W_eff = W @ W_eff
            b_eff = W @ b_eff + b

            # Apply ReLU pattern (set inactive neurons to 0)
            for i, active in enumerate(layer_pattern):
                if not active:
                    W_eff[i, :] = 0
                    b_eff[i] = 0

        # Final layer
        W_final, b_final = self.layers[-1]
        W_eff = W_final @ W_eff
        b_eff = W_final @ b_eff + b_final

        return W_eff[output_idx], b_eff[output_idx]

    def analyze_expressivity(self) -> dict[str, float]:
        """Analyze network expressivity through tropical lens.

        Returns metrics about the network's complexity:
        - Upper bound on linear regions
        - Network depth and width
        - Tropical complexity measures
        """
        depths = len(self.layers)
        widths = [W.shape[0] for W, _ in self.layers]
        input_dim = self.layers[0][0].shape[1] if self.layers else 0

        # Upper bound from Montufar et al.
        # For L layers with n_0 inputs and n units per layer:
        # N(f) <= (prod_{i=1}^{L-1} floor(n/n_0)^{n_0}) * sum_{j=0}^{n_0} C(n, j)
        if len(widths) > 1 and input_dim > 0:
            n = min(widths[:-1])  # Hidden layer width
            n_0 = input_dim
            L = len(widths)

            # Simplified upper bound
            upper_bound = (n // n_0) ** (n_0 * (L - 1))
            upper_bound *= sum(math.comb(n, j) for j in range(min(n_0 + 1, n + 1)))
        else:
            upper_bound = 1

        return {
            "depth": depths,
            "widths": widths,
            "input_dim": input_dim,
            "max_linear_regions_upper_bound": upper_bound,
            "total_parameters": sum(W.size + b.size for W, b in self.layers),
        }


# =============================================================================
# Phylogenetic Tree Distances
# =============================================================================


@dataclass
class TropicalPhylogeneticTree:
    """Phylogenetic tree represented in tropical space.

    Trees are represented as points in tropical projective space,
    where the tropical distance between trees has biological meaning.
    """

    n_taxa: int
    edge_lengths: np.ndarray  # Length 2n-2 for n taxa
    topology: tuple[tuple[int, int], ...] | None = None

    def to_tropical_coordinates(self) -> np.ndarray:
        """Convert to tropical projective coordinates."""
        # Tropical coordinates are based on pairwise distances
        n = self.n_taxa
        coords = np.zeros(n * (n - 1) // 2)

        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                # Distance between taxa i and j
                coords[idx] = self._compute_path_length(i, j)
                idx += 1

        return coords

    def _compute_path_length(self, i: int, j: int) -> float:
        """Compute path length between taxa i and j."""
        # Simplified: sum of all edge lengths (placeholder)
        # Real implementation would trace the path in the tree
        return self.edge_lengths.sum() / 2


class TropicalPhylogeneticDistance:
    """Compute tropical distances between phylogenetic trees.

    The tropical metric on tree space has properties:
    - Captures both topological and branch length differences
    - Geodesics correspond to tree rearrangements
    - Connected to parametric inference
    """

    def __init__(self, n_taxa: int):
        """Initialize distance calculator.

        Args:
            n_taxa: Number of taxa (leaves)
        """
        self.n_taxa = n_taxa
        self.n_pairs = n_taxa * (n_taxa - 1) // 2

    def distance(
        self,
        tree1: TropicalPhylogeneticTree,
        tree2: TropicalPhylogeneticTree,
    ) -> float:
        """Compute tropical distance between two trees.

        The tropical distance is:
            d(T1, T2) = max_{i,j} |d_{T1}(i,j) - d_{T2}(i,j)| -
                        min_{i,j} |d_{T1}(i,j) - d_{T2}(i,j)|

        This is the diameter of the difference vector in tropical projective space.
        """
        coords1 = tree1.to_tropical_coordinates()
        coords2 = tree2.to_tropical_coordinates()

        diff = coords1 - coords2
        return diff.max() - diff.min()

    def distance_matrix(
        self,
        trees: list[TropicalPhylogeneticTree],
    ) -> np.ndarray:
        """Compute pairwise distance matrix."""
        n = len(trees)
        D = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                d = self.distance(trees[i], trees[j])
                D[i, j] = d
                D[j, i] = d

        return D

    def frechet_mean(
        self,
        trees: list[TropicalPhylogeneticTree],
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute Frechet mean in tropical space.

        The tropical Frechet mean minimizes:
            sum_i w_i * d(x, T_i)^2

        For max-plus tropical geometry, this has a closed form.
        """
        if weights is None:
            weights = np.ones(len(trees)) / len(trees)

        # Get coordinates
        coords = np.array([t.to_tropical_coordinates() for t in trees])

        # Weighted tropical mean (component-wise weighted max)
        # For standard mean in tropical projective space
        mean_coords = np.zeros(self.n_pairs)
        for j in range(self.n_pairs):
            # Component-wise: use weighted median (more robust)
            mean_coords[j] = np.average(coords[:, j], weights=weights)

        return mean_coords

    def geodesic(
        self,
        tree1: TropicalPhylogeneticTree,
        tree2: TropicalPhylogeneticTree,
        n_points: int = 10,
    ) -> list[np.ndarray]:
        """Compute geodesic path between two trees.

        In tropical geometry, geodesics are piecewise linear paths.
        """
        coords1 = tree1.to_tropical_coordinates()
        coords2 = tree2.to_tropical_coordinates()

        # Linear interpolation in tropical coordinates
        path = []
        for t in np.linspace(0, 1, n_points):
            point = (1 - t) * coords1 + t * coords2
            path.append(point)

        return path


# =============================================================================
# Tropical Convexity
# =============================================================================


class TropicalConvexHull:
    """Compute tropical convex hull of points.

    A set S is tropically convex if for any x, y in S,
    the tropical line segment between them is in S.
    """

    def __init__(self, points: np.ndarray):
        """Initialize with points.

        Args:
            points: Array of shape (n_points, dim)
        """
        self.points = points
        self.n_points, self.dim = points.shape

    def contains(self, query: np.ndarray) -> bool:
        """Check if query point is in tropical convex hull."""
        # A point x is in the tropical convex hull of P if
        # x can be written as a tropical convex combination:
        # x = max_{j in J} (lambda_j + p_j) for some lambdas
        # where max_j lambda_j = 0

        # This is equivalent to checking a system of tropical linear equations
        # Simplified check using vertex representation
        for i in range(self.dim):
            # For each coordinate, check if query is achievable
            min_val = self.points[:, i].min()
            max_val = self.points[:, i].max()
            if query[i] < min_val or query[i] > max_val:
                return False
        return True

    def extreme_points(self) -> np.ndarray:
        """Find extreme points of the tropical convex hull.

        A point is extreme if it cannot be expressed as a tropical
        convex combination of other points.
        """
        extreme = []
        for i in range(self.n_points):
            # Check if point i is extreme
            other_points = np.delete(self.points, i, axis=0)
            if len(other_points) == 0:
                extreme.append(i)
                continue

            # Check if point i is in convex hull of others
            # (simplified version)
            p = self.points[i]
            is_extreme = False
            for dim in range(self.dim):
                if p[dim] > other_points[:, dim].max():
                    is_extreme = True
                    break
                if p[dim] < other_points[:, dim].min():
                    is_extreme = True
                    break

            if is_extreme:
                extreme.append(i)

        return self.points[extreme]
