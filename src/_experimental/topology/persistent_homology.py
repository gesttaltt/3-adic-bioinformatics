# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Persistent Homology module for topological data analysis.

This module implements persistent homology for protein structure analysis,
providing topological fingerprints that capture multi-scale structural features.

Key features:
- Vietoris-Rips filtration for point cloud analysis
- P-adic filtration integrating hierarchical structure
- Persistence diagram vectorization for ML pipelines
- Neural network encoder for end-to-end learning

References:
- Edelsbrunner & Harer (2010): Computational Topology
- Carlsson (2009): Topology and Data
- Adams et al. (2017): Persistence Images
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn


@dataclass
class PersistenceDiagram:
    """Representation of a persistence diagram.

    A persistence diagram encodes the birth and death times of topological
    features (connected components, loops, voids) across a filtration.

    Attributes:
        dimension: Homology dimension (0=components, 1=loops, 2=voids)
        birth: Array of birth times for each feature
        death: Array of death times for each feature
        generators: Optional generators (representative cycles)

    Example:
        >>> diagram = PersistenceDiagram(
        ...     dimension=1,
        ...     birth=np.array([0.0, 0.1, 0.2]),
        ...     death=np.array([0.5, 0.3, 0.8]),
        ... )
        >>> print(diagram.persistence)  # [0.5, 0.2, 0.6]
    """

    dimension: int
    birth: np.ndarray
    death: np.ndarray
    generators: list[np.ndarray] | None = None

    def __post_init__(self):
        """Validate and ensure arrays are numpy arrays."""
        self.birth = np.asarray(self.birth, dtype=np.float32)
        self.death = np.asarray(self.death, dtype=np.float32)
        if len(self.birth) != len(self.death):
            raise ValueError("birth and death arrays must have same length")

    def __len__(self) -> int:
        """Return number of features in diagram."""
        return len(self.birth)

    @property
    def persistence(self) -> np.ndarray:
        """Compute persistence (death - birth) for each feature."""
        return self.death - self.birth

    @property
    def midlife(self) -> np.ndarray:
        """Compute midlife ((birth + death) / 2) for each feature."""
        return (self.birth + self.death) / 2

    def filter_by_persistence(self, threshold: float) -> PersistenceDiagram:
        """Return diagram with only features persisting above threshold."""
        mask = self.persistence > threshold
        return PersistenceDiagram(
            dimension=self.dimension,
            birth=self.birth[mask],
            death=self.death[mask],
            generators=None,  # Generators would need filtering too
        )

    def to_tensor(self) -> torch.Tensor:
        """Convert to torch tensor of shape (n_features, 2)."""
        return torch.tensor(
            np.stack([self.birth, self.death], axis=1),
            dtype=torch.float32,
        )

    @classmethod
    def empty(cls, dimension: int = 0) -> PersistenceDiagram:
        """Create an empty persistence diagram."""
        return cls(
            dimension=dimension,
            birth=np.array([], dtype=np.float32),
            death=np.array([], dtype=np.float32),
        )

    def wasserstein_distance(self, other: PersistenceDiagram, p: int = 2) -> float:
        """Compute p-Wasserstein distance to another diagram.

        Uses optimal transport between diagrams, including matching
        to the diagonal.

        Args:
            other: Another persistence diagram
            p: Order of Wasserstein distance (default: 2)

        Returns:
            Wasserstein distance as float
        """
        if len(self) == 0 and len(other) == 0:
            return 0.0

        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            raise ImportError("scipy required for Wasserstein distance")

        # Create cost matrix including diagonal projections
        n, m = len(self), len(other)
        total = n + m

        cost = np.full((total, total), np.inf, dtype=np.float32)

        # Point-to-point costs
        for i in range(n):
            for j in range(m):
                cost[i, j] = (abs(self.birth[i] - other.birth[j]) ** p + abs(self.death[i] - other.death[j]) ** p) ** (
                    1 / p
                )

        # Point-to-diagonal costs (self points)
        for i in range(n):
            diag_cost = abs(self.death[i] - self.birth[i]) / (2 ** (1 / p))
            cost[i, m + i] = diag_cost  # Match to dummy

        # Point-to-diagonal costs (other points)
        for j in range(m):
            diag_cost = abs(other.death[j] - other.birth[j]) / (2 ** (1 / p))
            cost[n + j, j] = diag_cost

        # Diagonal-to-diagonal (zero cost)
        for i in range(n):
            for j in range(m):
                if i + m < total and j + n < total:
                    cost[m + i, n + j] = 0.0

        row_ind, col_ind = linear_sum_assignment(cost)
        return float(cost[row_ind, col_ind].sum())


@dataclass
class TopologicalFingerprint:
    """Multi-dimensional topological fingerprint.

    Combines persistence diagrams from multiple homology dimensions
    into a unified representation.

    Attributes:
        diagrams: Dict mapping dimension to PersistenceDiagram
        metadata: Optional metadata about the computation

    Example:
        >>> fingerprint = TopologicalFingerprint(diagrams={
        ...     0: diagram_h0,
        ...     1: diagram_h1,
        ... })
        >>> betti = fingerprint.betti_numbers(threshold=0.5)
    """

    diagrams: dict = field(default_factory=dict)
    metadata: dict | None = None

    def __getitem__(self, dimension: int) -> PersistenceDiagram:
        """Get persistence diagram for given dimension."""
        if dimension not in self.diagrams:
            return PersistenceDiagram.empty(dimension)
        return self.diagrams[dimension]

    def __len__(self) -> int:
        """Return number of dimensions with diagrams."""
        return len(self.diagrams)

    @property
    def max_dimension(self) -> int:
        """Return highest dimension with features."""
        if not self.diagrams:
            return -1
        return max(self.diagrams.keys())

    @property
    def total_features(self) -> int:
        """Return total number of topological features."""
        return sum(len(d) for d in self.diagrams.values())

    def betti_numbers(self, threshold: float = 0.0) -> dict:
        """Compute Betti numbers at given filtration value.

        Betti_k = number of k-dimensional holes that persist
        above the threshold.

        Args:
            threshold: Filtration threshold value

        Returns:
            Dict mapping dimension to Betti number
        """
        betti = {}
        for dim, diag in self.diagrams.items():
            # Count features that are "alive" at threshold
            alive = (diag.birth <= threshold) & (diag.death > threshold)
            betti[dim] = int(alive.sum())
        return betti


class RipsFiltration:
    """Vietoris-Rips complex filtration for point clouds.

    Constructs a Vietoris-Rips complex by connecting points within
    increasing distance thresholds, then computes persistent homology.

    Attributes:
        max_dimension: Maximum homology dimension to compute
        max_edge_length: Maximum edge length to consider
        n_threads: Number of threads for computation

    Example:
        >>> filt = RipsFiltration(max_dimension=2)
        >>> fingerprint = filt.build(protein_coordinates)
        >>> print(fingerprint.total_features)
    """

    def __init__(
        self,
        max_dimension: int = 2,
        max_edge_length: float = np.inf,
        n_threads: int = 1,
    ):
        """Initialize Rips filtration.

        Args:
            max_dimension: Maximum homology dimension (0, 1, 2, ...)
            max_edge_length: Maximum edge length threshold
            n_threads: Number of parallel threads
        """
        self.max_dimension = max_dimension
        self.max_edge_length = max_edge_length
        self.n_threads = n_threads
        self._backend = self._detect_backend()

    def _detect_backend(self) -> str:
        """Detect available TDA backend."""
        try:
            import ripser

            return "ripser"
        except ImportError:
            pass

        try:
            import gudhi

            return "gudhi"
        except ImportError:
            pass

        return "numpy"  # Fallback to basic implementation

    def build(
        self,
        points: np.ndarray | torch.Tensor,
    ) -> TopologicalFingerprint:
        """Build Rips complex and compute persistent homology.

        Args:
            points: Point cloud of shape (n_points, n_dims)

        Returns:
            TopologicalFingerprint with persistence diagrams
        """
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()

        points = np.asarray(points, dtype=np.float32)

        if self._backend == "ripser":
            return self._build_ripser(points)
        elif self._backend == "gudhi":
            return self._build_gudhi(points)
        else:
            return self._build_numpy(points)

    def _build_ripser(self, points: np.ndarray) -> TopologicalFingerprint:
        """Use ripser for efficient computation."""
        import ripser

        result = ripser.ripser(
            points,
            maxdim=self.max_dimension,
            thresh=self.max_edge_length,
        )

        diagrams = {}
        for dim, dgm in enumerate(result["dgms"]):
            if len(dgm) > 0:
                # Filter out infinite death times for visualization
                finite_mask = np.isfinite(dgm[:, 1])
                birth = dgm[:, 0]
                death = np.where(finite_mask, dgm[:, 1], dgm[:, 0] + 1.0)
                diagrams[dim] = PersistenceDiagram(
                    dimension=dim,
                    birth=birth,
                    death=death,
                )

        return TopologicalFingerprint(
            diagrams=diagrams,
            metadata={"backend": "ripser", "n_points": len(points)},
        )

    def _build_gudhi(self, points: np.ndarray) -> TopologicalFingerprint:
        """Use GUDHI for computation."""
        import gudhi

        rips_complex = gudhi.RipsComplex(
            points=points,
            max_edge_length=self.max_edge_length,
        )
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=self.max_dimension + 1)
        simplex_tree.compute_persistence()

        diagrams = {}
        for dim in range(self.max_dimension + 1):
            simplex_tree.persistence_pairs()
            # Filter by dimension
            dim_pairs = [(b, d) for (b, d) in simplex_tree.persistence() if b[0] == dim]
            if dim_pairs:
                birth = np.array([p[1][0] for p in dim_pairs], dtype=np.float32)
                death = np.array([p[1][1] for p in dim_pairs], dtype=np.float32)
                # Replace inf with large value
                death = np.where(np.isinf(death), birth + 1.0, death)
                diagrams[dim] = PersistenceDiagram(dimension=dim, birth=birth, death=death)

        return TopologicalFingerprint(
            diagrams=diagrams,
            metadata={"backend": "gudhi", "n_points": len(points)},
        )

    def _build_numpy(self, points: np.ndarray) -> TopologicalFingerprint:
        """Basic numpy implementation for H0 (connected components).

        This is a fallback when specialized libraries aren't available.
        Only computes H0 using a simplified union-find approach.
        """
        from scipy.spatial.distance import pdist, squareform

        n = len(points)
        if n == 0:
            return TopologicalFingerprint()

        # Compute pairwise distances
        dist_matrix = squareform(pdist(points)) if n > 1 else np.zeros((1, 1))

        # Get all unique edge lengths sorted
        edge_lengths = np.unique(dist_matrix[np.triu_indices(n, k=1)])
        edge_lengths = edge_lengths[edge_lengths <= self.max_edge_length]
        edge_lengths = np.sort(edge_lengths)

        # Track component births (all at 0) and deaths
        births = [0.0]  # One infinite component
        deaths = [np.inf]

        # Simple union-find for H0
        parent = list(range(n))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
                return True
            return False

        n_components = n
        for eps in edge_lengths:
            for i in range(n):
                for j in range(i + 1, n):
                    if dist_matrix[i, j] <= eps and union(i, j):
                        n_components -= 1
                        # Record death of a component
                        if n_components > 0:
                            births.append(0.0)
                            deaths.append(eps)

            if n_components == 1:
                break

        diagrams = {
            0: PersistenceDiagram(
                dimension=0,
                birth=np.array(births[1:], dtype=np.float32),  # Skip infinite
                death=np.array(deaths[1:], dtype=np.float32),
            )
        }

        return TopologicalFingerprint(
            diagrams=diagrams,
            metadata={"backend": "numpy", "n_points": n},
        )


class PAdicFiltration:
    """P-adic number filtration for hierarchical data.

    Constructs a filtration based on p-adic distances, which naturally
    captures hierarchical structure in data.

    The p-adic distance d_p(x, y) = p^(-v_p(x-y)) where v_p is the
    p-adic valuation.

    Args:
        prime: Prime base for p-adic numbers (default: 3)
        max_dimension: Maximum homology dimension
        max_valuation: Maximum valuation to consider

    Example:
        >>> filt = PAdicFiltration(prime=3)
        >>> fingerprint = filt.build([0, 1, 3, 9, 27])
    """

    def __init__(
        self,
        prime: int = 3,
        max_dimension: int = 1,
        max_valuation: int = 9,
    ):
        """Initialize p-adic filtration.

        Args:
            prime: Prime base for p-adic numbers
            max_dimension: Maximum homology dimension
            max_valuation: Maximum valuation to consider
        """
        self.prime = prime
        self.max_dimension = max_dimension
        self.max_valuation = max_valuation

    def _compute_valuation(self, n: int) -> int:
        """Compute p-adic valuation of integer n."""
        if n == 0:
            return self.max_valuation

        v = 0
        while n % self.prime == 0:
            n //= self.prime
            v += 1
        return min(v, self.max_valuation)

    def _padic_distance(self, i: int, j: int) -> float:
        """Compute p-adic distance between indices."""
        if i == j:
            return 0.0
        v = self._compute_valuation(abs(i - j))
        return float(self.prime ** (-v))

    def build(
        self,
        indices: np.ndarray | torch.Tensor | list[int],
    ) -> TopologicalFingerprint:
        """Build filtration from p-adic indices.

        Args:
            indices: Integer indices representing p-adic numbers

        Returns:
            TopologicalFingerprint computed from p-adic filtration
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().numpy()
        indices = np.asarray(indices, dtype=np.int64)

        n = len(indices)
        if n == 0:
            return TopologicalFingerprint()

        # Build distance matrix
        dist_matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                d = self._padic_distance(int(indices[i]), int(indices[j]))
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        # Use Rips filtration on precomputed distances
        return self._build_from_distance_matrix(dist_matrix)

    def _build_from_distance_matrix(self, dist_matrix: np.ndarray) -> TopologicalFingerprint:
        """Build filtration from precomputed distance matrix."""
        try:
            import ripser

            result = ripser.ripser(
                dist_matrix,
                maxdim=self.max_dimension,
                distance_matrix=True,
            )

            diagrams = {}
            for dim, dgm in enumerate(result["dgms"]):
                if len(dgm) > 0:
                    finite_mask = np.isfinite(dgm[:, 1])
                    birth = dgm[:, 0]
                    death = np.where(finite_mask, dgm[:, 1], birth + 0.1)
                    diagrams[dim] = PersistenceDiagram(dimension=dim, birth=birth, death=death)

            return TopologicalFingerprint(
                diagrams=diagrams,
                metadata={"backend": "ripser", "prime": self.prime},
            )
        except ImportError:
            # Fallback: simple H0 computation
            return self._h0_from_distance_matrix(dist_matrix)

    def _h0_from_distance_matrix(self, dist_matrix: np.ndarray) -> TopologicalFingerprint:
        """Compute H0 from distance matrix using union-find."""
        n = len(dist_matrix)
        if n == 0:
            return TopologicalFingerprint()

        # Get all unique distances sorted
        distances = np.unique(dist_matrix[np.triu_indices(n, k=1)])
        distances = np.sort(distances)

        births = []
        deaths = []

        parent = list(range(n))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
                return True
            return False

        n_components = n
        for eps in distances:
            for i in range(n):
                for j in range(i + 1, n):
                    if dist_matrix[i, j] <= eps and union(i, j):
                        n_components -= 1
                        births.append(0.0)
                        deaths.append(eps)

        if births:
            diagrams = {
                0: PersistenceDiagram(
                    dimension=0,
                    birth=np.array(births, dtype=np.float32),
                    death=np.array(deaths, dtype=np.float32),
                )
            }
        else:
            diagrams = {}

        return TopologicalFingerprint(
            diagrams=diagrams,
            metadata={"backend": "numpy", "prime": self.prime},
        )


class PersistenceVectorizer:
    """Convert persistence diagrams to fixed-size vectors.

    Supports multiple vectorization methods:
    - statistics: Summary statistics (mean, std, max persistence, etc.)
    - landscape: Piecewise linear persistence landscape functions
    - image: 2D persistence image histogram

    This enables using persistent homology features in standard ML pipelines.

    Args:
        method: Vectorization method ('statistics', 'landscape', 'image')
        resolution: Resolution for image/landscape methods
        sigma: Bandwidth for kernel density estimation
        dimensions: Which homology dimensions to include

    Example:
        >>> vectorizer = PersistenceVectorizer(method="statistics")
        >>> vector = vectorizer.transform(fingerprint)
        >>> print(vector.shape)  # (20,) for 2 dimensions
    """

    def __init__(
        self,
        method: str = "statistics",
        resolution: int = 50,
        sigma: float = 0.1,
        dimensions: list[int] = None,
    ):
        """Initialize vectorizer.

        Args:
            method: Vectorization method ('statistics', 'landscape', 'image')
            resolution: Resolution for image/landscape methods
            sigma: Bandwidth for kernel density estimation
            dimensions: Which homology dimensions to include
        """
        self.method = method
        self.resolution = resolution
        self.sigma = sigma
        self.dimensions = dimensions or [0, 1]

    def transform(self, fingerprint: TopologicalFingerprint) -> np.ndarray:
        """Transform fingerprint to vector.

        Args:
            fingerprint: TopologicalFingerprint to vectorize

        Returns:
            1D numpy array representation
        """
        if self.method == "statistics":
            return self._statistics_vector(fingerprint)
        elif self.method == "landscape":
            return self._landscape_vector(fingerprint)
        elif self.method == "image":
            return self._persistence_image(fingerprint)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _statistics_vector(self, fingerprint: TopologicalFingerprint) -> np.ndarray:
        """Compute summary statistics from diagrams.

        For each dimension, computes:
        - Number of features
        - Mean, std, max, sum of persistence
        - Mean, std of birth times
        - Mean, std of death times
        """
        stats = []
        for dim in self.dimensions:
            diag = fingerprint[dim]
            if len(diag) == 0:
                stats.extend([0.0] * 10)
            else:
                pers = diag.persistence
                stats.extend(
                    [
                        float(len(diag)),  # count
                        float(np.mean(pers)),  # mean persistence
                        float(np.std(pers)) if len(pers) > 1 else 0.0,
                        float(np.max(pers)),  # max persistence
                        float(np.sum(pers)),  # total persistence
                        float(np.mean(diag.birth)),  # mean birth
                        float(np.std(diag.birth)) if len(diag.birth) > 1 else 0.0,
                        float(np.mean(diag.death)),  # mean death
                        float(np.std(diag.death)) if len(diag.death) > 1 else 0.0,
                        float(np.mean(diag.midlife)),  # mean midlife
                    ]
                )

        return np.array(stats, dtype=np.float32)

    def _landscape_vector(self, fingerprint: TopologicalFingerprint) -> np.ndarray:
        """Compute persistence landscape representation.

        The k-th landscape function is the k-th largest value of
        the piecewise linear functions centered at each point.
        """
        vectors = []
        for dim in self.dimensions:
            diag = fingerprint[dim]
            if len(diag) == 0:
                vectors.append(np.zeros(self.resolution, dtype=np.float32))
                continue

            # Compute landscape at fixed grid
            max_death = diag.death.max() if len(diag) > 0 else 1.0
            t = np.linspace(0, max_death * 1.1, self.resolution)

            # For each point (b, d), the tent function is:
            # max(0, min(t - b, d - t))
            landscape = np.zeros(self.resolution, dtype=np.float32)
            for b, d in zip(diag.birth, diag.death, strict=False):
                tent = np.maximum(0, np.minimum(t - b, d - t))
                landscape = np.maximum(landscape, tent)

            vectors.append(landscape)

        return np.concatenate(vectors)

    def _persistence_image(self, fingerprint: TopologicalFingerprint) -> np.ndarray:
        """Compute persistence image representation.

        Creates a 2D histogram in birth-persistence coordinates,
        weighted by persistence and smoothed with Gaussian kernel.
        """
        vectors = []
        for dim in self.dimensions:
            diag = fingerprint[dim]
            if len(diag) == 0:
                vectors.append(np.zeros(self.resolution * self.resolution, dtype=np.float32))
                continue

            # Transform to birth-persistence coordinates
            birth = diag.birth
            pers = diag.persistence

            # Create grid
            b_min, b_max = 0, birth.max() + 1e-6 if len(birth) > 0 else 1
            p_min, p_max = 0, pers.max() + 1e-6 if len(pers) > 0 else 1

            b_grid = np.linspace(b_min, b_max, self.resolution)
            p_grid = np.linspace(p_min, p_max, self.resolution)

            # Compute weighted image
            image = np.zeros((self.resolution, self.resolution), dtype=np.float32)
            for b, p in zip(birth, pers, strict=False):
                # Weight by persistence
                weight = p

                # Gaussian kernel
                b_idx = np.argmin(np.abs(b_grid - b))
                p_idx = np.argmin(np.abs(p_grid - p))

                for i in range(max(0, b_idx - 2), min(self.resolution, b_idx + 3)):
                    for j in range(max(0, p_idx - 2), min(self.resolution, p_idx + 3)):
                        db = (b_grid[i] - b) / self.sigma
                        dp = (p_grid[j] - p) / self.sigma
                        image[i, j] += weight * np.exp(-(db**2 + dp**2) / 2)

            vectors.append(image.flatten())

        return np.concatenate(vectors)

    @property
    def output_dim(self) -> int:
        """Return output dimension based on method and settings."""
        n_dims = len(self.dimensions)
        if self.method == "statistics":
            return 10 * n_dims
        elif self.method == "landscape":
            return self.resolution * n_dims
        elif self.method == "image":
            return self.resolution * self.resolution * n_dims
        else:
            raise ValueError(f"Unknown method: {self.method}")


class ProteinTopologyEncoder(nn.Module):
    """Neural network encoder using persistent homology features.

    Computes topological fingerprints from protein coordinates and
    encodes them into a fixed-size embedding for downstream tasks.

    Architecture:
    1. Compute persistent homology from input coordinates
    2. Vectorize persistence diagrams
    3. Feed through MLP to produce embeddings

    Args:
        output_dim: Output embedding dimension
        hidden_dims: Hidden layer dimensions
        max_dimension: Maximum homology dimension
        vectorization: Vectorization method
        resolution: Resolution for vectorization
        use_padic: Whether to use p-adic filtration
        prime: Prime for p-adic filtration

    Example:
        >>> encoder = ProteinTopologyEncoder(output_dim=128)
        >>> embeddings = encoder(coordinates_batch)
    """

    def __init__(
        self,
        output_dim: int = 128,
        hidden_dims: list[int] = None,
        max_dimension: int = 1,
        vectorization: str = "statistics",
        resolution: int = 50,
        use_padic: bool = False,
        prime: int = 3,
    ):
        """Initialize topology encoder.

        Args:
            output_dim: Output embedding dimension
            hidden_dims: Hidden layer dimensions
            max_dimension: Maximum homology dimension
            vectorization: Vectorization method
            resolution: Resolution for vectorization
            use_padic: Whether to use p-adic filtration
            prime: Prime for p-adic filtration
        """
        super().__init__()

        self.output_dim = output_dim
        self.hidden_dims = hidden_dims or [256, 128]
        self.max_dimension = max_dimension
        self.vectorization = vectorization
        self.use_padic = use_padic

        # Filtration
        self.rips = RipsFiltration(max_dimension=max_dimension)
        if use_padic:
            self.padic = PAdicFiltration(prime=prime, max_dimension=max_dimension)

        # Vectorizer
        self.vectorizer = PersistenceVectorizer(
            method=vectorization,
            resolution=resolution,
            dimensions=list(range(max_dimension + 1)),
        )

        # MLP
        input_dim = self.vectorizer.output_dim
        if use_padic:
            input_dim *= 2  # Concatenate Rips and p-adic features

        layers = []
        dims = [input_dim] + self.hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        coordinates: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            coordinates: Protein coordinates (batch, n_atoms, 3)
            indices: Optional p-adic indices for each sample

        Returns:
            Embedding tensor of shape (batch, output_dim)
        """
        batch_size = coordinates.shape[0]
        device = coordinates.device

        embeddings = []
        for i in range(batch_size):
            coords = coordinates[i].detach().cpu().numpy()

            # Compute topological fingerprint
            fingerprint = self.rips.build(coords)
            vector = self.vectorizer.transform(fingerprint)

            if self.use_padic and indices is not None:
                idx = indices[i].detach().cpu().numpy()
                padic_fingerprint = self.padic.build(idx)
                padic_vector = self.vectorizer.transform(padic_fingerprint)
                vector = np.concatenate([vector, padic_vector])

            embeddings.append(torch.tensor(vector, dtype=torch.float32))

        embeddings = torch.stack(embeddings).to(device)
        return self.mlp(embeddings)

    def compute_fingerprint(self, coordinates: np.ndarray) -> TopologicalFingerprint:
        """Compute topological fingerprint without embedding.

        Useful for analysis and visualization.

        Args:
            coordinates: Protein coordinates

        Returns:
            TopologicalFingerprint
        """
        return self.rips.build(coordinates)
