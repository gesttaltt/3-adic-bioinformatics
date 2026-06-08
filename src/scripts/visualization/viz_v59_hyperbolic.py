# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Detailed v5.9 Hyperbolic Poincare Ball Geometry Analysis."""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.paths import CHECKPOINTS_DIR, VIZ_DIR

output_path = VIZ_DIR
output_path.mkdir(parents=True, exist_ok=True)

# Load v5.9 epoch 150
ckpt = torch.load(
    str(CHECKPOINTS_DIR / "v5_9" / "epoch_150.pt"),
    map_location="cpu",
    weights_only=False,
)

print("Generating v5.9 Hyperbolic Geometry Analysis...")

# Extract all metrics
cov_A = np.array(ckpt["coverage_A_history"]) / 19683 * 100
cov_B = np.array(ckpt["coverage_B_history"]) / 19683 * 100
corr_hyp = np.array(ckpt["correlation_history_hyp"])
corr_euc = np.array(ckpt["correlation_history_euc"])
rw = np.array(ckpt.get("ranking_weight_history", []))
H_A = np.array(ckpt.get("H_A_history", []))
H_B = np.array(ckpt.get("H_B_history", []))

epochs = range(len(cov_A))

# Create figure
fig = plt.figure(figsize=(18, 14))
gs = GridSpec(3, 3, figure=fig)

# Main correlation plot with both metrics
ax = fig.add_subplot(gs[0, :2])
ax.plot(
    epochs,
    corr_hyp,
    label="Hyperbolic (Poincare Ball)",
    color="#9b59b6",
    linewidth=2.5,
)
ax.plot(
    epochs,
    corr_euc,
    label="Euclidean",
    color="#3498db",
    linewidth=2,
    linestyle="--",
)
ax.fill_between(
    epochs,
    corr_hyp,
    corr_euc,
    alpha=0.2,
    color="#9b59b6",
    where=(corr_hyp > corr_euc),
    label="Hyp advantage",
)
ax.axhline(0.7, color="gold", linestyle="--", alpha=0.7, label="0.7 threshold")
ax.axhline(
    max(corr_hyp),
    color="green",
    linestyle=":",
    alpha=0.5,
    label=f"Max: {max(corr_hyp):.4f}",
)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("3-adic Distance Correlation (r)", fontsize=12)
ax.set_title(
    "v5.9: Hyperbolic vs Euclidean 3-adic Distance Correlation\n(Poincare Ball Geometry)",
    fontsize=14,
)
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
ax.set_ylim(0.55, 0.75)

# Best epoch annotation
best_epoch = np.argmax(corr_hyp)
ax.annotate(
    f"Peak: {max(corr_hyp):.4f}\nEpoch {best_epoch}",
    xy=(best_epoch, max(corr_hyp)),
    xytext=(best_epoch + 10, max(corr_hyp) - 0.02),
    arrowprops=dict(arrowstyle="->", color="green"),
    fontsize=10,
    color="green",
)

# Coverage metrics
ax = fig.add_subplot(gs[0, 2])
ax.plot(epochs, cov_A, label="Coverage A", color="#3498db", linewidth=2)
ax.plot(epochs, cov_B, label="Coverage B", color="#e74c3c", linewidth=2)
ax.fill_between(epochs, cov_A, cov_B, alpha=0.2)
ax.set_xlabel("Epoch")
ax.set_ylabel("Coverage (%)")
ax.set_title("Coverage Evolution")
ax.legend()
ax.grid(True, alpha=0.3)

# Ranking weight dynamics
ax = fig.add_subplot(gs[1, 0])
ax.plot(epochs, rw, color="#9b59b6", linewidth=2)
ax.fill_between(epochs, 0, rw, alpha=0.3, color="#9b59b6")
ax.axhline(1.0, color="red", linestyle="--", alpha=0.5, label="Max (1.0)")
ax.axhline(0.5, color="orange", linestyle="--", alpha=0.5, label="Balanced")
ax.set_xlabel("Epoch")
ax.set_ylabel("Ranking Weight")
ax.set_title("Dynamic Ranking Weight\n(StateNet v3 Metric-Aware)")
ax.legend()
ax.grid(True, alpha=0.3)

# Entropy dynamics
ax = fig.add_subplot(gs[1, 1])
if len(H_A) > 0:
    ax.plot(range(len(H_A)), H_A, label="H_A (VAE-A)", color="#3498db", linewidth=2)
    ax.plot(range(len(H_B)), H_B, label="H_B (VAE-B)", color="#e74c3c", linewidth=2)
    ax.fill_between(range(len(H_A)), H_A, H_B, alpha=0.2)
ax.set_xlabel("Epoch")
ax.set_ylabel("Latent Entropy")
ax.set_title("Entropy Evolution")
ax.legend()
ax.grid(True, alpha=0.3)

# Correlation improvement rate
ax = fig.add_subplot(gs[1, 2])
improvement = np.diff(corr_hyp)
smoothed = np.convolve(improvement, np.ones(5) / 5, mode="valid")
ax.plot(range(len(smoothed)), smoothed, color="#9b59b6", linewidth=1.5, alpha=0.7)
ax.axhline(0, color="gray", linestyle="-")
ax.fill_between(
    range(len(smoothed)),
    0,
    smoothed,
    where=(smoothed > 0),
    alpha=0.3,
    color="green",
)
ax.fill_between(
    range(len(smoothed)),
    0,
    smoothed,
    where=(smoothed <= 0),
    alpha=0.3,
    color="red",
)
ax.set_xlabel("Epoch")
ax.set_ylabel("d(Correlation)/d(Epoch)")
ax.set_title("Learning Rate (Smoothed)")
ax.grid(True, alpha=0.3)

# Phase analysis
ax = fig.add_subplot(gs[2, 0])
phase1 = range(0, 50)
phase2 = range(50, 100)
phase3 = range(100, 151)
ax.scatter(
    phase1,
    corr_hyp[:50],
    s=20,
    alpha=0.6,
    label="Phase 1 (0-50)",
    color="#3498db",
)
ax.scatter(
    phase2,
    corr_hyp[50:100],
    s=20,
    alpha=0.6,
    label="Phase 2 (50-100)",
    color="#e67e22",
)
ax.scatter(
    phase3,
    corr_hyp[100:],
    s=20,
    alpha=0.6,
    label="Phase 3 (100-150)",
    color="#2ecc71",
)
ax.set_xlabel("Epoch")
ax.set_ylabel("Hyperbolic Correlation")
ax.set_title("Training Phases")
ax.legend()
ax.grid(True, alpha=0.3)

# Distribution of correlation values
ax = fig.add_subplot(gs[2, 1])
ax.hist(
    corr_hyp,
    bins=30,
    alpha=0.7,
    color="#9b59b6",
    edgecolor="black",
    label="Hyperbolic",
)
ax.hist(
    corr_euc,
    bins=30,
    alpha=0.5,
    color="#3498db",
    edgecolor="black",
    label="Euclidean",
)
ax.axvline(
    np.mean(corr_hyp),
    color="#9b59b6",
    linestyle="--",
    linewidth=2,
    label=f"Hyp mean: {np.mean(corr_hyp):.4f}",
)
ax.axvline(
    np.mean(corr_euc),
    color="#3498db",
    linestyle="--",
    linewidth=2,
    label=f"Euc mean: {np.mean(corr_euc):.4f}",
)
ax.set_xlabel("Correlation")
ax.set_ylabel("Frequency")
ax.set_title("Correlation Distribution")
ax.legend(fontsize=8)

# Summary statistics
ax = fig.add_subplot(gs[2, 2])
ax.axis("off")
summary = f"""
v5.9 HYPERBOLIC GEOMETRY SUMMARY
================================

Model: Dual-VAE with Poincare Ball
Epochs: 150

CORRELATION METRICS
  Hyperbolic:
    Max:   {max(corr_hyp):.4f} (epoch {np.argmax(corr_hyp)})
    Final: {corr_hyp[-1]:.4f}
    Mean:  {np.mean(corr_hyp):.4f}
    Std:   {np.std(corr_hyp):.4f}

  Euclidean:
    Max:   {max(corr_euc):.4f} (epoch {np.argmax(corr_euc)})
    Final: {corr_euc[-1]:.4f}
    Mean:  {np.mean(corr_euc):.4f}

COVERAGE
  A: {cov_A[-1]:.2f}% (max: {max(cov_A):.2f}%)
  B: {cov_B[-1]:.2f}% (max: {max(cov_B):.2f}%)

INSIGHT
  Hyperbolic geometry provides
  {((max(corr_hyp) - max(corr_euc)) / max(corr_euc) * 100):+.1f}% improvement
  in peak 3-adic correlation,
  validating the ultrametric
  structure hypothesis.
"""
ax.text(
    0.1,
    0.95,
    summary,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
    fontfamily="monospace",
    bbox=dict(boxstyle="round", facecolor="#f0e68c", alpha=0.8),
)

plt.tight_layout()
plt.savefig(output_path / "v59_hyperbolic_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {output_path / 'v59_hyperbolic_analysis.png'}")
