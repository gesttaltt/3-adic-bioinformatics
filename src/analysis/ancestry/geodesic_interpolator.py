# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Geodesic Interpolation for Ancestral Sequence Reconstruction.

This module implements ancestral state reconstruction using geodesic
interpolation in hyperbolic space (Poincare ball model).

Key Concepts:
1. Geodesic Interpolation: The shortest path between two points
   in hyperbolic space passes through their common ancestor
2. Frechet Mean: The geometric center of multiple points minimizes
   sum of squared geodesic distances
3. Weighted Midpoints: Account for branch lengths in phylogenetic trees

Mathematical Foundation:
In the Poincare ball, the geodesic from x to y is:
    gamma(t) = x (+) (t (*) ((-x) (+) y))

where:
- (+) is Mobius addition
- (*) is scalar Mobius multiplication
- (-x) is Mobius negation

The midpoint (t=0.5) approximates the ancestral state.

For phylogenetic trees with branch lengths, we use weighted Frechet
means that account for evolutionary distances.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from src.geometry.poincare import (
    get_manifold,
    mobius_add,
    poincare_distance,
    project_to_poincare,
)


@dataclass
class ReconstructionConfig:
    """Configuration for ancestral reconstruction."""

    curvature: float = 1.0
    max_norm: float = 0.95
    frechet_max_iterations: int = 100
    frechet_tolerance: float = 1e-6
    use_branch_lengths: bool = True
    uncertainty_method: str = "bootstrap"  # 'bootstrap', 'laplace', 'none'
    n_bootstrap_samples: int = 100


@dataclass
class TreeNode:
    """Node in a phylogenetic tree."""

    name: str
    embedding: torch.Tensor | None = None  # Latent representation
    children: list[TreeNode] = field(default_factory=list)
    parent: TreeNode | None = None
    branch_length: float = 1.0
    is_leaf: bool = True
    sequence: str | None = None


@dataclass
class AncestralState:
    """Reconstructed ancestral state with uncertainty."""

    embedding: torch.Tensor  # Mean ancestral embedding
    uncertainty: torch.Tensor  # Uncertainty (covariance or confidence)
    confidence: float  # Overall confidence score
    decoded_sequence: str | None = None
    support_values: dict[int, float] | None = None  # Per-position support


@dataclass
class AncestralNode:
    """Complete ancestral node with reconstruction results."""

    node: TreeNode
    state: AncestralState
    children_states: list[AncestralState] = field(default_factory=list)
    reconstruction_method: str = "geodesic_midpoint"


class GeodesicInterpolator(nn.Module):
    """Interpolate along geodesics in hyperbolic space.

    Provides methods for:
    1. Pairwise geodesic interpolation
    2. Multi-point Frechet mean computation
    3. Weighted averages with branch lengths
    4. Uncertainty quantification
    """

    def __init__(self, config: ReconstructionConfig | None = None):
        """Initialize geodesic interpolator.

        Args:
            config: Configuration for reconstruction
        """
        super().__init__()
        self.config = config or ReconstructionConfig()
        self.manifold = get_manifold(self.config.curvature)

    def geodesic_point(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t: float = 0.5,
    ) -> torch.Tensor:
        """Compute point along geodesic from x to y.

        Args:
            x: Start point (batch, dim)
            y: End point (batch, dim)
            t: Interpolation parameter [0, 1], t=0.5 gives midpoint

        Returns:
            Point on geodesic at parameter t
        """
        # gamma(t) = x (+) (t (*) ((-x) (+) y))
        # Step 1: Compute (-x) (+) y (direction from x to y)
        neg_x = -x  # Mobius negation is just negation in Poincare ball
        direction = mobius_add(neg_x, y, c=self.config.curvature)

        # Step 2: Scalar Mobius multiplication: t (*) v
        # For Poincare ball: t (*) v = tanh(t * arctanh(||v||)) * v / ||v||
        scaled_direction = self._mobius_scalar_mul(direction, t)

        # Step 3: Add to x
        result = mobius_add(x, scaled_direction, c=self.config.curvature)

        return project_to_poincare(result, self.config.max_norm, self.config.curvature)

    def _mobius_scalar_mul(
        self,
        v: torch.Tensor,
        r: float,
    ) -> torch.Tensor:
        """Mobius scalar multiplication.

        r (*) v = tanh(r * arctanh(sqrt(c)||v||)) * v / (sqrt(c)||v||)

        Args:
            v: Vector in Poincare ball
            r: Scalar multiplier

        Returns:
            Scaled vector
        """
        c = self.config.curvature
        sqrt_c = math.sqrt(c)

        v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-10)

        # arctanh(sqrt(c) * ||v||)
        scaled_norm = (sqrt_c * v_norm).clamp(max=1 - 1e-5)
        arctanh_norm = torch.arctanh(scaled_norm)

        # tanh(r * arctanh_norm)
        new_norm = torch.tanh(r * arctanh_norm)

        # Scale vector
        return v * (new_norm / (sqrt_c * v_norm))

    def geodesic_midpoint(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute geodesic midpoint (simplified ancestral approximation).

        Args:
            x: First descendant embedding
            y: Second descendant embedding

        Returns:
            Midpoint on geodesic (approximate ancestor)
        """
        return self.geodesic_point(x, y, t=0.5)

    def weighted_midpoint(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        weight_x: float,
        weight_y: float,
    ) -> torch.Tensor:
        """Compute weighted geodesic midpoint.

        Useful when branch lengths differ between descendants.

        Args:
            x: First descendant embedding
            y: Second descendant embedding
            weight_x: Weight for x (e.g., inverse branch length)
            weight_y: Weight for y (e.g., inverse branch length)

        Returns:
            Weighted midpoint
        """
        # Normalize weights
        total = weight_x + weight_y
        t = weight_y / total  # t=0 gives x, t=1 gives y

        return self.geodesic_point(x, y, t)

    def frechet_mean(
        self,
        points: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute Frechet mean (Karcher/Riemannian center of mass).

        The Frechet mean minimizes:
            sum_i w_i * d(m, x_i)^2

        Uses iterative gradient descent in the tangent space.

        Args:
            points: Points to average (n, dim)
            weights: Optional weights (n,)

        Returns:
            Frechet mean point
        """
        n = points.size(0)
        device = points.device

        weights = torch.ones(n, device=device) / n if weights is None else weights / weights.sum()

        # Initialize with weighted Euclidean mean (projected)
        mean = (weights.unsqueeze(-1) * points).sum(dim=0)
        mean = project_to_poincare(mean.unsqueeze(0), self.config.max_norm, self.config.curvature).squeeze(0)

        # Iterative refinement
        for _ in range(self.config.frechet_max_iterations):
            # Compute tangent vectors at mean
            tangent_vecs = self.manifold.logmap(mean.unsqueeze(0), points)

            # Weighted average in tangent space
            weighted_tangent = (weights.unsqueeze(-1) * tangent_vecs).sum(dim=0)

            # Check convergence
            if torch.norm(weighted_tangent) < self.config.frechet_tolerance:
                break

            # Update mean via exponential map
            mean = self.manifold.expmap(mean.unsqueeze(0), weighted_tangent.unsqueeze(0)).squeeze(0)
            mean = project_to_poincare(mean.unsqueeze(0), self.config.max_norm, self.config.curvature).squeeze(0)

        return mean

    def geodesic_variance(
        self,
        mean: torch.Tensor,
        points: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute geodesic variance around mean.

        Args:
            mean: Center point
            points: Data points (n, dim)
            weights: Optional weights

        Returns:
            Variance (scalar)
        """
        n = points.size(0)
        device = points.device

        if weights is None:
            weights = torch.ones(n, device=device) / n

        # Compute squared distances
        dists_sq = (
            poincare_distance(
                mean.unsqueeze(0).expand(n, -1),
                points,
                c=self.config.curvature,
            )
            ** 2
        )

        return (weights * dists_sq).sum()

    def interpolation_path(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        n_steps: int = 10,
    ) -> torch.Tensor:
        """Generate points along geodesic path.

        Args:
            x: Start point
            y: End point
            n_steps: Number of interpolation steps

        Returns:
            Path points (n_steps, dim)
        """
        t_values = torch.linspace(0, 1, n_steps, device=x.device)
        path = []

        for t in t_values:
            point = self.geodesic_point(x.unsqueeze(0), y.unsqueeze(0), t.item())
            path.append(point.squeeze(0))

        return torch.stack(path)


class PhylogeneticReconstructor(nn.Module):
    """Reconstruct ancestral states on a phylogenetic tree.

    Uses geodesic interpolation and Frechet means to reconstruct
    internal node states from leaf embeddings.
    """

    def __init__(
        self,
        config: ReconstructionConfig | None = None,
        decoder: nn.Module | None = None,
    ):
        """Initialize phylogenetic reconstructor.

        Args:
            config: Configuration for reconstruction
            decoder: Optional decoder to convert embeddings to sequences
        """
        super().__init__()
        self.config = config or ReconstructionConfig()
        self.interpolator = GeodesicInterpolator(self.config)
        self.decoder = decoder

    def reconstruct_internal_node(
        self,
        children: list[TreeNode],
    ) -> AncestralState:
        """Reconstruct internal node from its children.

        Args:
            children: List of child nodes with embeddings

        Returns:
            Reconstructed ancestral state
        """
        # Collect child embeddings and weights
        embeddings = []
        weights = []

        for child in children:
            if child.embedding is not None:
                embeddings.append(child.embedding)
                # Weight by inverse branch length (closer = more informative)
                if self.config.use_branch_lengths:
                    weights.append(1.0 / (child.branch_length + 1e-6))
                else:
                    weights.append(1.0)

        if len(embeddings) == 0:
            raise ValueError("No embeddings available for reconstruction")

        embeddings = torch.stack(embeddings)
        weights = torch.tensor(weights, device=embeddings.device)

        # Compute Frechet mean
        if len(embeddings) == 2:
            # Use weighted midpoint for binary trees
            ancestor = self.interpolator.weighted_midpoint(
                embeddings[0].unsqueeze(0),
                embeddings[1].unsqueeze(0),
                weights[0].item(),
                weights[1].item(),
            ).squeeze(0)
        else:
            # Use Frechet mean for polytomies
            ancestor = self.interpolator.frechet_mean(embeddings, weights)

        # Compute uncertainty
        variance = self.interpolator.geodesic_variance(ancestor, embeddings, weights)
        uncertainty = torch.sqrt(variance).expand_as(ancestor)
        confidence = 1.0 / (1.0 + variance.item())

        # Decode if decoder available
        decoded = None
        if self.decoder is not None:
            with torch.no_grad():
                decoded = self.decoder(ancestor.unsqueeze(0))
                if isinstance(decoded, torch.Tensor):
                    decoded = self._tensor_to_sequence(decoded)

        return AncestralState(
            embedding=ancestor,
            uncertainty=uncertainty,
            confidence=confidence,
            decoded_sequence=decoded,
        )

    def reconstruct_tree(
        self,
        root: TreeNode,
    ) -> dict[str, AncestralNode]:
        """Reconstruct all internal nodes in a tree.

        Performs bottom-up reconstruction from leaves to root.

        Args:
            root: Root node of the tree

        Returns:
            Dictionary mapping node names to reconstructed states
        """
        results = {}
        self._reconstruct_recursive(root, results)
        return results

    def _reconstruct_recursive(
        self,
        node: TreeNode,
        results: dict[str, AncestralNode],
    ) -> AncestralState:
        """Recursively reconstruct from leaves up.

        Args:
            node: Current node
            results: Dictionary to store results

        Returns:
            State of current node
        """
        if node.is_leaf:
            # Leaf nodes have known embeddings
            if node.embedding is None:
                raise ValueError(f"Leaf node {node.name} has no embedding")

            state = AncestralState(
                embedding=node.embedding,
                uncertainty=torch.zeros_like(node.embedding),
                confidence=1.0,
                decoded_sequence=node.sequence,
            )
            return state

        # Reconstruct children first
        child_states = []
        for child in node.children:
            child_state = self._reconstruct_recursive(child, results)
            child_states.append(child_state)

            # Update child embedding for reconstruction
            child.embedding = child_state.embedding

        # Reconstruct this node
        state = self.reconstruct_internal_node(node.children)

        # Store result
        results[node.name] = AncestralNode(
            node=node,
            state=state,
            children_states=child_states,
            reconstruction_method="frechet_mean" if len(node.children) > 2 else "geodesic_midpoint",
        )

        return state

    def bootstrap_confidence(
        self,
        node: TreeNode,
        n_samples: int = 100,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute bootstrap confidence for ancestral reconstruction.

        Args:
            node: Node to compute confidence for
            n_samples: Number of bootstrap samples

        Returns:
            Tuple of (mean_embedding, std_embedding)
        """
        if node.is_leaf or len(node.children) == 0:
            if node.embedding is None:
                raise ValueError(f"Node {node.name} has no embedding")
            return node.embedding, torch.zeros_like(node.embedding)

        embeddings = torch.stack([c.embedding for c in node.children if c.embedding is not None])
        n_children = embeddings.size(0)

        bootstrap_ancestors = []
        for _ in range(n_samples):
            # Sample children with replacement
            indices = torch.randint(0, n_children, (n_children,))
            sampled = embeddings[indices]

            # Compute Frechet mean
            ancestor = self.interpolator.frechet_mean(sampled)
            bootstrap_ancestors.append(ancestor)

        bootstrap_stack = torch.stack(bootstrap_ancestors)

        return bootstrap_stack.mean(dim=0), bootstrap_stack.std(dim=0)

    def _tensor_to_sequence(self, tensor: torch.Tensor) -> str:
        """Convert decoder output tensor to sequence string.

        Args:
            tensor: Decoder output

        Returns:
            Sequence string
        """
        # Assume tensor is logits (batch, seq_len, vocab_size)
        if tensor.dim() == 3:
            indices = tensor.argmax(dim=-1)[0]  # (seq_len,)
        else:
            indices = tensor.argmax(dim=-1)

        # Map to amino acids (simplified)
        aa_list = "ARNDCEQGHILKMFPSTWYV"
        return "".join(aa_list[int(i) % len(aa_list)] for i in indices)

    def compute_ancestral_uncertainty(
        self,
        node: TreeNode,
        method: str = "laplace",
    ) -> torch.Tensor:
        """Compute uncertainty in ancestral reconstruction.

        Args:
            node: Node to compute uncertainty for
            method: 'laplace' for Laplace approximation, 'fisher' for Fisher info

        Returns:
            Uncertainty tensor (covariance diagonal or full matrix)
        """
        if method == "laplace":
            # Use curvature of Frechet objective as uncertainty
            if node.embedding is None:
                raise ValueError("Node has no embedding")

            child_embeddings = [c.embedding for c in node.children if c.embedding is not None]
            if not child_embeddings:
                return torch.zeros_like(node.embedding)

            embeddings = torch.stack(child_embeddings)

            # Compute Hessian approximation (sum of squared distances Hessian)
            # For hyperbolic space, curvature varies with position
            dists = poincare_distance(
                node.embedding.unsqueeze(0).expand(len(child_embeddings), -1),
                embeddings,
                c=self.config.curvature,
            )

            # Uncertainty scales with mean distance and inversely with n_children
            mean_dist = dists.mean()
            uncertainty = mean_dist / math.sqrt(len(child_embeddings))

            return uncertainty * torch.ones_like(node.embedding)

        elif method == "fisher":
            # Fisher information based uncertainty
            # Requires model gradients
            raise NotImplementedError("Fisher uncertainty requires model gradients")

        else:
            raise ValueError(f"Unknown uncertainty method: {method}")

    def path_to_ancestor(
        self,
        descendant: TreeNode,
        ancestor: TreeNode,
        n_steps: int = 10,
    ) -> torch.Tensor:
        """Generate evolutionary path from descendant to ancestor.

        Args:
            descendant: Descendant node
            ancestor: Ancestor node
            n_steps: Number of interpolation steps

        Returns:
            Path embeddings (n_steps, dim)
        """
        if descendant.embedding is None or ancestor.embedding is None:
            raise ValueError("Both nodes must have embeddings")

        return self.interpolator.interpolation_path(
            descendant.embedding,
            ancestor.embedding,
            n_steps,
        )


def reconstruct_mrca(
    embeddings: torch.Tensor,
    weights: torch.Tensor | None = None,
    curvature: float = 1.0,
) -> torch.Tensor:
    """Reconstruct Most Recent Common Ancestor from leaf embeddings.

    Simple convenience function for quick MRCA estimation.

    Args:
        embeddings: Leaf embeddings (n, dim)
        weights: Optional weights (e.g., inverse branch lengths)
        curvature: Poincare ball curvature

    Returns:
        MRCA embedding
    """
    config = ReconstructionConfig(curvature=curvature)
    interpolator = GeodesicInterpolator(config)
    return interpolator.frechet_mean(embeddings, weights)


def evolutionary_distance_matrix(
    embeddings: torch.Tensor,
    curvature: float = 1.0,
) -> torch.Tensor:
    """Compute pairwise evolutionary distances from embeddings.

    Uses hyperbolic distance as proxy for evolutionary distance.

    Args:
        embeddings: Sequence embeddings (n, dim)
        curvature: Poincare ball curvature

    Returns:
        Distance matrix (n, n)
    """
    from src.geometry.poincare import poincare_distance_matrix

    return poincare_distance_matrix(embeddings, c=curvature)


__all__ = [
    "ReconstructionConfig",
    "TreeNode",
    "AncestralState",
    "AncestralNode",
    "GeodesicInterpolator",
    "PhylogeneticReconstructor",
    "reconstruct_mrca",
    "evolutionary_distance_matrix",
]
