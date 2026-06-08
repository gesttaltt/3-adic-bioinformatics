# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Manifold Visualization Module.

This module provides plotting utilities for visualizing embeddings on
various manifolds, particularly the Poincare ball (hyperbolic space).

Key Features:
- 2D/3D Poincare disk visualization
- Geodesic path plotting
- Cluster visualization with decision boundaries
- Distance matrix heatmaps

Usage:
    from src.visualization.plots.manifold import (
        plot_poincare_disk,
        plot_geodesics,
        plot_embedding_clusters,
    )
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from src.visualization.core.base import create_figure, despine
from src.visualization.styles.palettes import SEMANTIC, TOLVIBRANT, get_categorical_cmap


def plot_poincare_disk(
    embeddings: np.ndarray,
    labels: np.ndarray | None = None,
    colors: Sequence[str] | None = None,
    title: str = "Poincare Disk Embedding",
    ax: Axes | None = None,
    figsize: tuple[float, float] = (8, 8),
    show_boundary: bool = True,
    show_origin: bool = True,
    marker_size: float = 50,
    alpha: float = 0.7,
    cmap: str | Colormap | None = None,
) -> tuple[Figure, Axes]:
    """Plot embeddings on the Poincare disk (2D hyperbolic space).

    Args:
        embeddings: Points on Poincare disk, shape (N, 2)
        labels: Optional cluster labels for coloring, shape (N,)
        colors: Optional explicit colors for each point
        title: Plot title
        ax: Existing axes to plot on
        figsize: Figure size if creating new figure
        show_boundary: Whether to show the unit circle boundary
        show_origin: Whether to mark the origin
        marker_size: Size of scatter markers
        alpha: Transparency of markers
        cmap: Colormap for labels (default: tolvibrant)

    Returns:
        Figure and Axes objects
    """
    if ax is None:
        fig, ax = create_figure(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Draw unit circle boundary
    if show_boundary:
        boundary = Circle((0, 0), 1, fill=False, edgecolor=SEMANTIC.grid, linewidth=2, linestyle="--")
        ax.add_patch(boundary)

    # Draw origin
    if show_origin:
        ax.plot(0, 0, "k+", markersize=10, mew=2)

    # Determine colors
    if colors is not None:
        c = colors
    elif labels is not None:
        if cmap is None:
            cmap = get_categorical_cmap("tolvibrant", n_colors=len(np.unique(labels)))
        c = [cmap(int(label)) for label in labels]
    else:
        c = SEMANTIC.primary

    # Plot embeddings
    ax.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c=c,
        s=marker_size,
        alpha=alpha,
        edgecolors="white",
        linewidths=0.5,
    )

    # Set equal aspect ratio and limits
    ax.set_aspect("equal")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_title(title)

    despine(ax)

    return fig, ax


def plot_geodesics(
    start_points: np.ndarray,
    end_points: np.ndarray,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (8, 8),
    n_steps: int = 50,
    curvature: float = 1.0,
    linewidth: float = 1.5,
    color: str = SEMANTIC.primary,
    alpha: float = 0.6,
    show_disk: bool = True,
) -> tuple[Figure, Axes]:
    """Plot geodesics (shortest paths) on the Poincare disk.

    Args:
        start_points: Starting points, shape (N, 2)
        end_points: Ending points, shape (N, 2)
        ax: Existing axes to plot on
        figsize: Figure size if creating new figure
        n_steps: Number of interpolation steps per geodesic
        curvature: Hyperbolic curvature parameter
        linewidth: Width of geodesic lines
        color: Line color
        alpha: Line transparency
        show_disk: Whether to show the Poincare disk boundary

    Returns:
        Figure and Axes objects
    """
    if ax is None:
        fig, ax = create_figure(figsize=figsize)
    else:
        fig = ax.get_figure()

    if show_disk:
        boundary = Circle((0, 0), 1, fill=False, edgecolor=SEMANTIC.grid, linewidth=2, linestyle="--")
        ax.add_patch(boundary)

    # Compute geodesic paths
    # For the Poincare disk, geodesics are arcs of circles perpendicular to the boundary
    # Here we use linear interpolation in the tangent space as an approximation
    for start, end in zip(start_points, end_points, strict=False):
        t = np.linspace(0, 1, n_steps)

        # Linear interpolation (approximation)
        # For exact geodesics, use geoopt.manifolds.PoincareBall.geodesic
        path = start[None, :] * (1 - t[:, None]) + end[None, :] * t[:, None]

        # Clamp to ball
        norms = np.linalg.norm(path, axis=1, keepdims=True)
        path = np.where(norms > 0.99, path * 0.99 / norms, path)

        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=linewidth, alpha=alpha)

    ax.set_aspect("equal")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

    despine(ax)

    return fig, ax


def plot_embedding_clusters(
    embeddings: np.ndarray,
    labels: np.ndarray,
    cluster_names: list[str] | None = None,
    title: str = "Embedding Clusters",
    ax: Axes | None = None,
    figsize: tuple[float, float] = (10, 8),
    show_centroids: bool = True,
    show_ellipses: bool = True,
    ellipse_std: float = 2.0,
    alpha: float = 0.6,
) -> tuple[Figure, Axes]:
    """Plot clustered embeddings with optional ellipse boundaries.

    Args:
        embeddings: Points, shape (N, 2)
        labels: Cluster labels, shape (N,)
        cluster_names: Optional names for each cluster
        title: Plot title
        ax: Existing axes
        figsize: Figure size
        show_centroids: Whether to show cluster centroids
        show_ellipses: Whether to show covariance ellipses
        ellipse_std: Number of standard deviations for ellipse
        alpha: Point transparency

    Returns:
        Figure and Axes objects
    """
    from matplotlib.patches import Ellipse

    if ax is None:
        fig, ax = create_figure(figsize=figsize)
    else:
        fig = ax.get_figure()

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    colors = TOLVIBRANT[:n_clusters]

    for i, label in enumerate(unique_labels):
        mask = labels == label
        cluster_points = embeddings[mask]

        # Scatter plot
        cluster_name = cluster_names[i] if cluster_names else f"Cluster {label}"
        ax.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            c=colors[i],
            label=cluster_name,
            alpha=alpha,
            s=50,
            edgecolors="white",
            linewidths=0.5,
        )

        if len(cluster_points) < 2:
            continue

        # Centroid
        centroid = cluster_points.mean(axis=0)

        if show_centroids:
            ax.plot(
                centroid[0],
                centroid[1],
                "o",
                color=colors[i],
                markersize=12,
                markeredgecolor="black",
                markeredgewidth=2,
            )

        # Covariance ellipse
        if show_ellipses and len(cluster_points) > 2:
            cov = np.cov(cluster_points.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            # Sort by eigenvalue
            order = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]

            # Ellipse parameters
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            width = 2 * ellipse_std * np.sqrt(eigenvalues[0])
            height = 2 * ellipse_std * np.sqrt(eigenvalues[1])

            ellipse = Ellipse(
                xy=centroid,
                width=width,
                height=height,
                angle=angle,
                facecolor=colors[i],
                alpha=0.2,
                edgecolor=colors[i],
                linewidth=2,
            )
            ax.add_patch(ellipse)

    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_title(title)
    ax.legend(loc="best")

    despine(ax)

    return fig, ax


def plot_distance_heatmap(
    distance_matrix: np.ndarray,
    labels: list[str] | None = None,
    title: str = "Pairwise Distance Matrix",
    ax: Axes | None = None,
    figsize: tuple[float, float] = (10, 8),
    cmap: str = "viridis",
    show_values: bool = False,
    value_format: str = ".2f",
) -> tuple[Figure, Axes]:
    """Plot heatmap of pairwise distances.

    Args:
        distance_matrix: Square distance matrix, shape (N, N)
        labels: Optional labels for each row/column
        title: Plot title
        ax: Existing axes
        figsize: Figure size
        cmap: Colormap name
        show_values: Whether to show numeric values in cells
        value_format: Format string for values

    Returns:
        Figure and Axes objects
    """
    if ax is None:
        fig, ax = create_figure(figsize=figsize)
    else:
        fig = ax.get_figure()

    im = ax.imshow(distance_matrix, cmap=cmap, aspect="equal")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Distance")

    # Labels
    if labels is not None:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)

    # Show values in cells
    if show_values and distance_matrix.shape[0] <= 20:
        for i in range(distance_matrix.shape[0]):
            for j in range(distance_matrix.shape[1]):
                value = distance_matrix[i, j]
                color = "white" if value > distance_matrix.max() / 2 else "black"
                ax.text(j, i, f"{value:{value_format}}", ha="center", va="center", color=color, fontsize=8)

    ax.set_title(title)

    return fig, ax


def plot_radial_distribution(
    embeddings: np.ndarray,
    labels: np.ndarray | None = None,
    title: str = "Radial Distribution",
    ax: Axes | None = None,
    figsize: tuple[float, float] = (10, 6),
    n_bins: int = 30,
    use_hyperbolic: bool = True,
    curvature: float = 1.0,
) -> tuple[Figure, Axes]:
    """Plot distribution of embedding distances from origin.

    Useful for understanding how embeddings fill the Poincare ball.

    Args:
        embeddings: Points, shape (N, D)
        labels: Optional cluster labels for separate distributions
        title: Plot title
        ax: Existing axes
        figsize: Figure size
        n_bins: Number of histogram bins
        use_hyperbolic: If True, compute hyperbolic distance (V5.12.2)
        curvature: Hyperbolic curvature parameter

    Returns:
        Figure and Axes objects
    """
    if ax is None:
        fig, ax = create_figure(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Compute distances from origin
    # V5.12.2: Use hyperbolic distance for Poincare ball embeddings
    euclidean_norms = np.linalg.norm(embeddings, axis=1)
    if use_hyperbolic:
        # Hyperbolic distance from origin: d_H = 2 * arctanh(sqrt(c) * ||x||)
        sqrt_c = np.sqrt(curvature)
        clamped_norms = np.clip(euclidean_norms * sqrt_c, 0, 0.999)
        distances = 2.0 * np.arctanh(clamped_norms) / sqrt_c
    else:
        distances = euclidean_norms

    if labels is None:
        ax.hist(distances, bins=n_bins, edgecolor="white", alpha=0.7, color=SEMANTIC.primary)
    else:
        unique_labels = np.unique(labels)
        colors = TOLVIBRANT[: len(unique_labels)]

        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.hist(
                distances[mask],
                bins=n_bins,
                alpha=0.6,
                color=colors[i],
                label=f"Cluster {label}",
                edgecolor="white",
            )

        ax.legend()

    ax.axvline(x=1.0, color="red", linestyle="--", linewidth=2, label="Ball boundary")
    ax.set_xlabel("Distance from Origin")
    ax.set_ylabel("Count")
    ax.set_title(title)

    despine(ax)

    return fig, ax


__all__ = [
    "plot_poincare_disk",
    "plot_geodesics",
    "plot_embedding_clusters",
    "plot_distance_heatmap",
    "plot_radial_distribution",
]
