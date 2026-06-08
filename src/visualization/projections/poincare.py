# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Poincare Ball Projection Visualization.

This module provides projection utilities for visualizing high-dimensional
hyperbolic embeddings on the 2D Poincare disk.

Key Features:
- Multiple projection methods (PCA, UMAP, t-SNE, geodesic)
- Interactive 3D Poincare ball visualization
- Animation support for trajectory visualization
- Hierarchical structure extraction

Usage:
    from src.visualization.projections.poincare import (
        project_to_2d_poincare,
        plot_3d_poincare_ball,
        animate_embedding_trajectory,
    )
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

logger = logging.getLogger(__name__)

from src.visualization.core.base import create_figure, despine
from src.visualization.styles.palettes import SEMANTIC, TOLVIBRANT


def hyperbolic_distance_from_origin_np(embeddings: np.ndarray, c: float = 1.0) -> np.ndarray:
    """V5.12.2: Compute hyperbolic distance from origin for numpy arrays.

    Args:
        embeddings: Points on Poincare ball, shape (N, D)
        c: Curvature parameter

    Returns:
        Hyperbolic distances, shape (N,)
    """
    sqrt_c = np.sqrt(c)
    euclidean_norms = np.linalg.norm(embeddings, axis=1)
    # Clamp to avoid arctanh(1) = infinity
    clamped = np.clip(euclidean_norms * sqrt_c, 0, 0.999)
    return 2.0 * np.arctanh(clamped) / sqrt_c


def project_to_2d_poincare(
    embeddings: np.ndarray,
    method: str = "pca",
    preserve_norms: bool = True,
    **kwargs,
) -> np.ndarray:
    """Project high-dimensional Poincare embeddings to 2D disk.

    Args:
        embeddings: Points on Poincare ball, shape (N, D)
        method: Projection method ('pca', 'umap', 'geodesic', 'random')
        preserve_norms: Whether to preserve radial distances from origin
        **kwargs: Method-specific arguments

    Returns:
        2D embeddings on Poincare disk, shape (N, 2)
    """
    n_samples, n_dims = embeddings.shape

    if n_dims == 2:
        return embeddings

    # Compute original norms
    original_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

    if method == "pca":
        # PCA projection
        centered = embeddings - embeddings.mean(axis=0)
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        projected = embeddings @ Vt[:2].T

    elif method == "umap":
        # UMAP projection (requires umap-learn)
        try:
            import umap

            reducer = umap.UMAP(n_components=2, metric="euclidean", **kwargs)
            projected = reducer.fit_transform(embeddings)
        except ImportError:
            logger.warning("UMAP not installed, falling back to PCA")
            return project_to_2d_poincare(embeddings, method="pca", preserve_norms=preserve_norms)

    elif method == "geodesic":
        # Project along geodesic to origin (radial preservation)
        # Take first 2 dims but scale to preserve geodesic distances
        projected = embeddings[:, :2].copy()

    elif method == "random":
        # Random 2D projection
        rng = np.random.default_rng(kwargs.get("seed", 42))
        projection_matrix = rng.standard_normal((n_dims, 2))
        projection_matrix, _ = np.linalg.qr(projection_matrix)
        projected = embeddings @ projection_matrix

    else:
        raise ValueError(f"Unknown projection method: {method}")

    # Normalize to unit disk
    max_norm = np.max(np.linalg.norm(projected, axis=1))
    if max_norm > 0:
        projected = projected / (max_norm * 1.05)

    # Optionally restore original radial structure
    if preserve_norms:
        projected_norms = np.linalg.norm(projected, axis=1, keepdims=True)
        # Scale to match original Poincare ball norms
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(projected_norms > 1e-8, original_norms / projected_norms, 1.0)
            # Clamp to ball
            scale = np.minimum(scale, 0.99 / projected_norms)
            projected = projected * scale

    return projected


def plot_3d_poincare_ball(
    embeddings: np.ndarray,
    labels: np.ndarray | None = None,
    title: str = "3D Poincare Ball",
    figsize: tuple[float, float] = (10, 10),
    show_surface: bool = True,
    surface_alpha: float = 0.1,
    elevation: float = 20,
    azimuth: float = 45,
    marker_size: float = 30,
    use_hyperbolic: bool = True,
    curvature: float = 1.0,
) -> tuple[Figure, Axes3D]:
    """Plot embeddings in 3D Poincare ball.

    Args:
        embeddings: 3D points on Poincare ball, shape (N, 3)
        labels: Optional cluster labels
        title: Plot title
        figsize: Figure size
        show_surface: Whether to show translucent ball surface
        surface_alpha: Transparency of ball surface
        elevation: Camera elevation angle
        azimuth: Camera azimuth angle
        marker_size: Size of scatter markers
        use_hyperbolic: V5.12.2 - Use hyperbolic distance for coloring (default True)
        curvature: Hyperbolic curvature parameter

    Returns:
        Figure and 3D Axes objects
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Draw unit sphere surface
    if show_surface:
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=surface_alpha, color="gray", linewidth=0)

    # Plot embeddings
    if labels is not None:
        unique_labels = np.unique(labels)
        colors = TOLVIBRANT[: len(unique_labels)]

        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                embeddings[mask, 0],
                embeddings[mask, 1],
                embeddings[mask, 2],
                c=colors[i],
                s=marker_size,
                label=f"Cluster {label}",
                alpha=0.7,
            )
        ax.legend()
    else:
        # V5.12.2: Color by distance from origin (hyperbolic or Euclidean)
        if use_hyperbolic:
            norms = hyperbolic_distance_from_origin_np(embeddings, c=curvature)
        else:
            norms = np.linalg.norm(embeddings, axis=1)
        scatter = ax.scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            embeddings[:, 2],
            c=norms,
            cmap="viridis",
            s=marker_size,
            alpha=0.7,
        )
        label = "Hyperbolic distance from origin" if use_hyperbolic else "Euclidean distance from origin"
        fig.colorbar(scatter, ax=ax, label=label, shrink=0.6)

    # Draw coordinate axes
    ax.plot([-1, 1], [0, 0], [0, 0], "k--", alpha=0.3)
    ax.plot([0, 0], [-1, 1], [0, 0], "k--", alpha=0.3)
    ax.plot([0, 0], [0, 0], [-1, 1], "k--", alpha=0.3)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    # Set equal aspect ratio
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 1.1)

    ax.view_init(elev=elevation, azim=azimuth)

    return fig, ax


def plot_hierarchy_tree(
    embeddings: np.ndarray,
    parent_indices: np.ndarray | None = None,
    node_labels: list[str] | None = None,
    title: str = "Hierarchical Structure",
    ax: Axes | None = None,
    figsize: tuple[float, float] = (12, 10),
    max_depth: int | None = None,
    use_hyperbolic: bool = True,
    curvature: float = 1.0,
) -> tuple[Figure, Axes]:
    """Plot hierarchical tree structure on Poincare disk.

    Visualizes parent-child relationships by drawing edges from parent
    nodes (closer to origin) to child nodes (closer to boundary).

    Args:
        embeddings: 2D Poincare disk embeddings, shape (N, 2)
        parent_indices: Index of parent for each node (-1 for roots)
        node_labels: Optional labels for each node
        title: Plot title
        ax: Existing axes
        figsize: Figure size
        max_depth: Maximum hierarchy depth to display
        use_hyperbolic: V5.12.2 - Use hyperbolic distance for depth coloring (default True)
        curvature: Hyperbolic curvature parameter

    Returns:
        Figure and Axes objects
    """
    from matplotlib.patches import Circle

    if ax is None:
        fig, ax = create_figure(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Draw Poincare disk boundary
    boundary = Circle((0, 0), 1, fill=False, edgecolor=SEMANTIC.grid, linewidth=2, linestyle="--")
    ax.add_patch(boundary)

    # Draw edges if parent_indices provided
    if parent_indices is not None:
        for i, parent_idx in enumerate(parent_indices):
            if parent_idx >= 0 and parent_idx < len(embeddings):
                ax.plot(
                    [embeddings[parent_idx, 0], embeddings[i, 0]],
                    [embeddings[parent_idx, 1], embeddings[i, 1]],
                    color=SEMANTIC.grid,
                    alpha=0.5,
                    linewidth=1,
                    zorder=1,
                )

    # V5.12.2: Color by depth (hyperbolic or Euclidean distance from origin)
    if use_hyperbolic:
        norms = hyperbolic_distance_from_origin_np(embeddings, c=curvature)
    else:
        norms = np.linalg.norm(embeddings, axis=1)
    scatter = ax.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c=norms,
        cmap="viridis",
        s=100,
        alpha=0.8,
        edgecolors="white",
        linewidths=0.5,
        zorder=2,
    )

    label = "Hierarchy Depth (hyperbolic distance)" if use_hyperbolic else "Hierarchy Depth (Euclidean distance)"
    fig.colorbar(scatter, ax=ax, label=label)

    # Add node labels if provided
    if node_labels is not None:
        for i, (x, y) in enumerate(embeddings):
            ax.annotate(
                node_labels[i],
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.7,
            )

    ax.set_aspect("equal")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_title(title)

    despine(ax)

    return fig, ax


def compute_hyperbolic_centroids(
    embeddings: np.ndarray,
    labels: np.ndarray,
    curvature: float = 1.0,
) -> np.ndarray:
    """Compute hyperbolic centroids (Frechet means) for clusters.

    Args:
        embeddings: Points on Poincare ball, shape (N, D)
        labels: Cluster labels, shape (N,)
        curvature: Hyperbolic curvature parameter

    Returns:
        Centroids for each cluster, shape (K, D)
    """
    unique_labels = np.unique(labels)
    centroids = np.zeros((len(unique_labels), embeddings.shape[1]))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        cluster_points = embeddings[mask]

        # Einstein midpoint (hyperbolic centroid approximation)
        # For exact computation, use iterative Frechet mean algorithm
        gamma = 1 / np.sqrt(1 - curvature * np.sum(cluster_points**2, axis=1, keepdims=True))
        weighted_sum = np.sum(gamma * cluster_points, axis=0)
        gamma_sum = np.sum(gamma)

        centroid = weighted_sum / gamma_sum

        # Project back to ball if needed
        norm = np.linalg.norm(centroid)
        if norm >= 1.0:
            centroid = centroid * 0.99 / norm

        centroids[i] = centroid

    return centroids


def create_poincare_animation_data(
    trajectory: np.ndarray,
    n_frames: int = 100,
) -> dict[str, np.ndarray]:
    """Prepare animation data for embedding trajectory.

    Args:
        trajectory: Time series of embeddings, shape (T, N, D)
        n_frames: Number of animation frames

    Returns:
        Dictionary with frame data
    """
    T, N, D = trajectory.shape

    # Sample frames evenly
    frame_indices = np.linspace(0, T - 1, n_frames, dtype=int)

    return {
        "frames": trajectory[frame_indices],
        "times": frame_indices,
        "n_frames": n_frames,
        "n_points": N,
        "n_dims": D,
    }


__all__ = [
    "project_to_2d_poincare",
    "plot_3d_poincare_ball",
    "plot_hierarchy_tree",
    "compute_hyperbolic_centroids",
    "create_poincare_animation_data",
]
