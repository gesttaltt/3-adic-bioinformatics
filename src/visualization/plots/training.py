# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Training Metrics Visualization Module.

This module provides plotting utilities for visualizing training progress,
loss curves, and optimization metrics.

Key Features:
- Multi-loss training curve visualization
- Learning rate scheduling plots
- Gradient flow analysis
- Validation metric tracking

Usage:
    from src.visualization.plots.training import (
        plot_training_curves,
        plot_loss_components,
        plot_gradient_norms,
    )
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.visualization.core.base import create_figure, despine
from src.visualization.styles.palettes import SEMANTIC, TOLVIBRANT


def plot_training_curves(
    history: dict[str, list[float]],
    metrics: list[str] | None = None,
    title: str = "Training Progress",
    ax: Axes | None = None,
    figsize: tuple[float, float] = (12, 6),
    show_best: bool = True,
    smooth_factor: float = 0.0,
    log_scale: bool = False,
) -> tuple[Figure, Axes]:
    """Plot training curves for multiple metrics.

    Args:
        history: Dictionary mapping metric names to lists of values
        metrics: Specific metrics to plot (None = all)
        title: Plot title
        ax: Existing axes
        figsize: Figure size
        show_best: Whether to mark the best value
        smooth_factor: Exponential smoothing factor (0 = no smoothing)
        log_scale: Whether to use log scale for y-axis

    Returns:
        Figure and Axes objects
    """
    if ax is None:
        fig, ax = create_figure(figsize=figsize)
    else:
        fig = ax.get_figure()

    if metrics is None:
        metrics = list(history.keys())

    colors = TOLVIBRANT[: len(metrics)]

    for i, metric in enumerate(metrics):
        if metric not in history:
            continue

        values = np.array(history[metric])
        epochs = np.arange(1, len(values) + 1)

        # Apply smoothing if requested
        if smooth_factor > 0:
            smoothed = np.zeros_like(values)
            smoothed[0] = values[0]
            for j in range(1, len(values)):
                smoothed[j] = smooth_factor * smoothed[j - 1] + (1 - smooth_factor) * values[j]
            ax.plot(epochs, smoothed, color=colors[i], linewidth=2, label=metric)
            ax.plot(epochs, values, color=colors[i], alpha=0.3, linewidth=1)
        else:
            ax.plot(epochs, values, color=colors[i], linewidth=2, label=metric)

        # Mark best value
        if show_best:
            best_idx = np.argmin(values) if "loss" in metric.lower() else np.argmax(values)
            ax.scatter(
                epochs[best_idx], values[best_idx], color=colors[i], s=100, zorder=5, marker="*", edgecolors="black"
            )

    if log_scale:
        ax.set_yscale("log")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend(loc="best")

    despine(ax)

    return fig, ax


def plot_loss_components(
    loss_history: dict[str, list[float]],
    title: str = "Loss Component Breakdown",
    figsize: tuple[float, float] = (12, 6),
    stacked: bool = False,
) -> tuple[Figure, Axes]:
    """Plot stacked or overlaid loss components.

    Args:
        loss_history: Dictionary mapping loss names to value lists
        title: Plot title
        figsize: Figure size
        stacked: Whether to create a stacked area plot

    Returns:
        Figure and Axes objects
    """
    fig, ax = create_figure(figsize=figsize)

    loss_names = list(loss_history.keys())
    epochs = np.arange(1, len(list(loss_history.values())[0]) + 1)
    colors = TOLVIBRANT[: len(loss_names)]

    if stacked:
        # Stack area plot
        values = np.array([loss_history[name] for name in loss_names])
        ax.stackplot(epochs, values, labels=loss_names, colors=colors, alpha=0.7)
    else:
        # Overlaid line plot
        for i, name in enumerate(loss_names):
            ax.plot(epochs, loss_history[name], color=colors[i], linewidth=2, label=name)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend(loc="upper right")

    despine(ax)

    return fig, ax


def plot_gradient_norms(
    gradient_history: dict[str, list[float]],
    title: str = "Gradient Norm Evolution",
    ax: Axes | None = None,
    figsize: tuple[float, float] = (12, 6),
    clip_threshold: float | None = None,
) -> tuple[Figure, Axes]:
    """Plot gradient norm evolution during training.

    Args:
        gradient_history: Dictionary mapping layer names to gradient norm lists
        title: Plot title
        ax: Existing axes
        figsize: Figure size
        clip_threshold: Optional gradient clipping threshold to visualize

    Returns:
        Figure and Axes objects
    """
    if ax is None:
        fig, ax = create_figure(figsize=figsize)
    else:
        fig = ax.get_figure()

    layer_names = list(gradient_history.keys())
    colors = TOLVIBRANT[: len(layer_names)]

    for i, layer in enumerate(layer_names):
        norms = gradient_history[layer]
        steps = np.arange(1, len(norms) + 1)
        ax.plot(steps, norms, color=colors[i], alpha=0.7, linewidth=1.5, label=layer)

    if clip_threshold is not None:
        ax.axhline(
            y=clip_threshold, color="red", linestyle="--", linewidth=2, label=f"Clip threshold ({clip_threshold})"
        )

    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient Norm")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.set_yscale("log")

    despine(ax)

    return fig, ax


def plot_learning_rate(
    lr_history: list[float] | dict[str, list[float]],
    title: str = "Learning Rate Schedule",
    ax: Axes | None = None,
    figsize: tuple[float, float] = (10, 5),
) -> tuple[Figure, Axes]:
    """Plot learning rate schedule.

    Args:
        lr_history: List of LR values or dict of LR types to values
        title: Plot title
        ax: Existing axes
        figsize: Figure size

    Returns:
        Figure and Axes objects
    """
    if ax is None:
        fig, ax = create_figure(figsize=figsize)
    else:
        fig = ax.get_figure()

    if isinstance(lr_history, list):
        lr_history = {"lr": lr_history}

    colors = TOLVIBRANT[: len(lr_history)]

    for i, (name, values) in enumerate(lr_history.items()):
        steps = np.arange(1, len(values) + 1)
        ax.plot(steps, values, color=colors[i], linewidth=2, label=name)

    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title(title)
    ax.legend()
    ax.set_yscale("log")

    despine(ax)

    return fig, ax


def plot_train_val_comparison(
    train_metrics: dict[str, list[float]],
    val_metrics: dict[str, list[float]],
    metric_name: str = "loss",
    title: str | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> tuple[Figure, Axes]:
    """Plot training vs validation metric comparison.

    Args:
        train_metrics: Training metrics history
        val_metrics: Validation metrics history
        metric_name: Metric to compare
        title: Plot title
        ax: Existing axes
        figsize: Figure size

    Returns:
        Figure and Axes objects
    """
    if ax is None:
        fig, ax = create_figure(figsize=figsize)
    else:
        fig = ax.get_figure()

    if title is None:
        title = f"Train vs Validation: {metric_name}"

    train_vals = train_metrics.get(metric_name, [])
    val_vals = val_metrics.get(metric_name, [])

    epochs_train = np.arange(1, len(train_vals) + 1)
    epochs_val = np.arange(1, len(val_vals) + 1)

    ax.plot(epochs_train, train_vals, color=SEMANTIC.primary, linewidth=2, label="Train")
    ax.plot(epochs_val, val_vals, color=SEMANTIC.risk_high, linewidth=2, label="Validation")

    # Highlight overfitting region
    if len(train_vals) == len(val_vals) and len(train_vals) > 0:
        train_arr = np.array(train_vals)
        val_arr = np.array(val_vals)
        gap = val_arr - train_arr

        if np.any(gap > 0):
            ax.fill_between(
                epochs_train,
                train_arr,
                val_arr,
                where=gap > 0,
                alpha=0.2,
                color=SEMANTIC.risk_high,
                label="Overfitting gap",
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_title(title)
    ax.legend()

    despine(ax)

    return fig, ax


def plot_parameter_histogram(
    parameters: dict[str, np.ndarray],
    title: str = "Parameter Distribution",
    figsize: tuple[float, float] = (12, 8),
    n_cols: int = 3,
    n_bins: int = 50,
) -> Figure:
    """Plot histograms of parameter values across layers.

    Args:
        parameters: Dictionary mapping layer names to parameter arrays
        title: Overall title
        figsize: Figure size
        n_cols: Number of columns in subplot grid
        n_bins: Number of histogram bins

    Returns:
        Figure object
    """
    n_params = len(parameters)
    n_rows = (n_params + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()

    for i, (name, values) in enumerate(parameters.items()):
        ax = axes[i]
        values_flat = values.flatten()

        ax.hist(values_flat, bins=n_bins, edgecolor="white", alpha=0.7, color=TOLVIBRANT[i % len(TOLVIBRANT)])
        ax.axvline(x=0, color="red", linestyle="--", alpha=0.5)

        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")

        despine(ax)

    # Hide unused subplots
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    return fig


def create_training_dashboard(
    history: dict[str, list[float]],
    figsize: tuple[float, float] = (16, 12),
) -> Figure:
    """Create comprehensive training dashboard.

    Args:
        history: Complete training history with all metrics

    Returns:
        Figure with multiple subplots
    """
    fig = plt.figure(figsize=figsize)

    # Layout: 2x2 grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. Main loss curve
    ax1 = fig.add_subplot(gs[0, 0])
    loss_keys = [k for k in history if "loss" in k.lower()]
    if loss_keys:
        plot_training_curves(history, metrics=loss_keys, ax=ax1, title="Loss Curves")

    # 2. Non-loss metrics
    ax2 = fig.add_subplot(gs[0, 1])
    other_keys = [k for k in history if "loss" not in k.lower() and "lr" not in k.lower()]
    if other_keys:
        plot_training_curves(history, metrics=other_keys, ax=ax2, title="Other Metrics")

    # 3. Learning rate
    ax3 = fig.add_subplot(gs[1, 0])
    lr_keys = [k for k in history if "lr" in k.lower()]
    if lr_keys:
        lr_dict = {k: history[k] for k in lr_keys}
        plot_learning_rate(lr_dict, ax=ax3)

    # 4. Final epoch summary (bar chart)
    ax4 = fig.add_subplot(gs[1, 1])
    if history:
        final_values = {k: v[-1] for k, v in history.items() if v}
        names = list(final_values.keys())
        values = list(final_values.values())

        ax4.barh(names, values, color=TOLVIBRANT[: len(names)])
        ax4.set_xlabel("Final Value")
        ax4.set_title("Final Epoch Metrics")
        despine(ax4)

    fig.suptitle("Training Dashboard", fontsize=16, y=1.02)

    return fig


__all__ = [
    "plot_training_curves",
    "plot_loss_components",
    "plot_gradient_norms",
    "plot_learning_rate",
    "plot_train_val_comparison",
    "plot_parameter_histogram",
    "create_training_dashboard",
]
