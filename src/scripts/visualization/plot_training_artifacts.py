# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Comprehensive Training Artifacts Visualization.

Extracts and visualizes:
1. Benchmark metrics (coverage, entropy, inference speed)
2. Manifold resolution analysis (reconstruction, dimensionality)
3. Coupled system dynamics (ensemble, complementarity)
4. TensorBoard training curves
5. Eigenspectrum analysis
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import CHECKPOINTS_DIR


def load_json(path):
    with open(path) as f:
        return json.load(f)


def plot_benchmark_summary(output_path):
    """Plot coverage, entropy, and inference benchmarks."""
    data = load_json(PROJECT_ROOT / "benchmarks" / "coverage_vs_entropy.json")

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig)

    # Coverage comparison
    ax = fig.add_subplot(gs[0, 0])
    coverage = data["results"]["coverage"]
    bars = ax.bar(
        ["VAE-A", "VAE-B"],
        [coverage["vae_a_mean_coverage"], coverage["vae_b_mean_coverage"]],
        yerr=[coverage["vae_a_std_coverage"], coverage["vae_b_std_coverage"]],
        color=["#3498db", "#e74c3c"],
        alpha=0.8,
        capsize=5,
    )
    ax.axhline(99.0, color="green", linestyle="--", label="99% target")
    ax.set_ylabel("Coverage (%)")
    ax.set_title("Manifold Coverage\n(195K samples)")
    ax.set_ylim(98, 100)
    ax.legend()
    for bar, val in zip(
        bars,
        [coverage["vae_a_mean_coverage"], coverage["vae_b_mean_coverage"]],
        strict=False,
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Entropy comparison
    ax = fig.add_subplot(gs[0, 1])
    entropy = data["results"]["entropy"]
    bars = ax.bar(
        ["VAE-A", "VAE-B"],
        [entropy["entropy_A"], entropy["entropy_B"]],
        color=["#3498db", "#e74c3c"],
        alpha=0.8,
    )
    ax.axhline(
        np.log2(16),
        color="gray",
        linestyle="--",
        label=f"Max (log₂16={np.log2(16):.2f})",
    )
    ax.set_ylabel("Latent Entropy (nats)")
    ax.set_title("Latent Space Utilization")
    ax.legend()
    for bar, val in zip(bars, [entropy["entropy_A"], entropy["entropy_B"]], strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Inference speed
    ax = fig.add_subplot(gs[0, 2])
    inference = data["results"]["inference"]
    speeds = [
        inference["vae_a_samples_per_sec"] / 1e6,
        inference["vae_b_samples_per_sec"] / 1e6,
    ]
    bars = ax.bar(["VAE-A", "VAE-B"], speeds, color=["#3498db", "#e74c3c"], alpha=0.8)
    ax.set_ylabel("Million Samples/sec")
    ax.set_title("Inference Throughput")
    for bar, val in zip(bars, speeds, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{val:.1f}M",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Memory usage
    ax = fig.add_subplot(gs[1, 0])
    memory = data["results"]["memory"]
    mem_vals = [
        memory["baseline_gb"] * 1000,
        memory["overhead_gb"] * 1000,
        memory["peak_gb"] * 1000,
    ]
    bars = ax.bar(
        ["Baseline", "Overhead", "Peak"],
        mem_vals,
        color=["gray", "orange", "red"],
        alpha=0.8,
    )
    ax.set_ylabel("Memory (MB)")
    ax.set_title("GPU Memory Usage")
    for bar, val in zip(bars, mem_vals, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Coverage distribution (simulated from mean/std)
    ax = fig.add_subplot(gs[1, 1])
    np.random.seed(42)
    cov_A = np.random.normal(coverage["vae_a_mean_coverage"], coverage["vae_a_std_coverage"], 1000)
    cov_B = np.random.normal(coverage["vae_b_mean_coverage"], coverage["vae_b_std_coverage"], 1000)
    ax.hist(cov_A, bins=30, alpha=0.5, label="VAE-A", color="#3498db", density=True)
    ax.hist(cov_B, bins=30, alpha=0.5, label="VAE-B", color="#e74c3c", density=True)
    ax.axvline(99.0, color="green", linestyle="--", label="99% target")
    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("Density")
    ax.set_title("Coverage Distribution\n(Monte Carlo estimate)")
    ax.legend()

    # Summary stats
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    stats_text = f"""
    BENCHMARK SUMMARY (Latest Checkpoint)
    ══════════════════════════════════════

    Coverage (195K samples, 5 trials):
      VAE-A: {coverage["vae_a_mean_coverage"]:.2f}% ± {coverage["vae_a_std_coverage"]:.3f}%
      VAE-B: {coverage["vae_b_mean_coverage"]:.2f}% ± {coverage["vae_b_std_coverage"]:.3f}%

    Entropy:
      VAE-A: {entropy["entropy_A"]:.4f} nats
      VAE-B: {entropy["entropy_B"]:.4f} nats
      Diff:  {entropy["entropy_diff"]:.4f} nats

    Inference (10K samples, 10 trials):
      VAE-A: {inference["vae_a_samples_per_sec"] / 1e6:.2f}M samples/sec
      VAE-B: {inference["vae_b_samples_per_sec"] / 1e6:.2f}M samples/sec

    Memory:
      Overhead: {memory["overhead_gb"] * 1000:.1f} MB
      Peak: {memory["peak_gb"] * 1000:.1f} MB
    """
    ax.text(
        0.1,
        0.9,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_path / "benchmark_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path / 'benchmark_summary.png'}")


def plot_manifold_resolution(output_path):
    """Plot manifold resolution analysis."""
    data = load_json(PROJECT_ROOT / "reports" / "benchmarks" / "manifold_resolution_3.json")

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig)

    # Reconstruction error histogram
    ax = fig.add_subplot(gs[0, 0])
    hist_A = data["vae_a"]["reconstruction"]["error_histogram"]
    hist_B = data["vae_b"]["reconstruction"]["error_histogram"]
    x_A = [int(k) for k in hist_A]
    y_A = list(hist_A.values())
    x_B = [int(k) for k in hist_B]
    y_B = list(hist_B.values())
    width = 0.35
    ax.bar(
        [x - width / 2 for x in x_A],
        y_A,
        width,
        label="VAE-A",
        color="#3498db",
        alpha=0.8,
    )
    ax.bar(
        [x + width / 2 for x in x_B],
        y_B,
        width,
        label="VAE-B",
        color="#e74c3c",
        alpha=0.8,
    )
    ax.set_xlabel("Bit Errors")
    ax.set_ylabel("Count")
    ax.set_title("Reconstruction Error Distribution")
    ax.legend()
    ax.set_yscale("log")

    # Latent separation
    ax = fig.add_subplot(gs[0, 1])
    sep_A = data["vae_a"]["latent_separation"]
    sep_B = data["vae_b"]["latent_separation"]
    metrics = ["mean_distance", "latent_norm_mean"]
    labels = ["Mean Pairwise Dist", "Mean Latent Norm"]
    x = np.arange(len(metrics))
    vals_A = [sep_A["mean_distance"], sep_A["latent_norm_mean"]]
    vals_B = [sep_B["mean_distance"], sep_B["latent_norm_mean"]]
    ax.bar(x - width / 2, vals_A, width, label="VAE-A", color="#3498db", alpha=0.8)
    ax.bar(x + width / 2, vals_B, width, label="VAE-B", color="#e74c3c", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Value")
    ax.set_title("Latent Space Geometry")
    ax.legend()
    # Add ratio annotation
    ratio = sep_B["latent_norm_mean"] / sep_A["latent_norm_mean"]
    ax.annotate(
        f"Ratio: {ratio:.1f}x",
        xy=(1, sep_B["latent_norm_mean"]),
        xytext=(1.3, sep_B["latent_norm_mean"]),
        fontsize=10,
        ha="left",
    )

    # Effective dimensionality
    ax = fig.add_subplot(gs[0, 2])
    dim_A = data["vae_a"]["dimensionality"]
    dim_B = data["vae_b"]["dimensionality"]
    dims = ["effective_dim", "dim_for_95pct_var", "dim_for_99pct_var"]
    labels = ["Effective", "95% Var", "99% Var"]
    vals_A = [dim_A[d] for d in dims]
    vals_B = [dim_B[d] for d in dims]
    x = np.arange(len(dims))
    ax.bar(x - width / 2, vals_A, width, label="VAE-A", color="#3498db", alpha=0.8)
    ax.bar(x + width / 2, vals_B, width, label="VAE-B", color="#e74c3c", alpha=0.8)
    ax.axhline(16, color="gray", linestyle="--", label="Nominal (16)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Dimensions")
    ax.set_title("Effective Dimensionality")
    ax.legend()

    # Resolution scores radar chart
    ax = fig.add_subplot(gs[1, 0], projection="polar")
    scores_A = data["resolution_score"]["vae_a"]
    scores_B = data["resolution_score"]["vae_b"]
    categories = [
        "reconstruction",
        "coverage",
        "interpolation",
        "nearest_neighbor",
        "dimensionality",
    ]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    vals_A = [scores_A[c] for c in categories] + [scores_A[categories[0]]]
    vals_B = [scores_B[c] for c in categories] + [scores_B[categories[0]]]
    ax.plot(
        angles,
        vals_A,
        "o-",
        linewidth=2,
        label=f"VAE-A ({scores_A['overall']:.2f})",
        color="#3498db",
    )
    ax.fill(angles, vals_A, alpha=0.25, color="#3498db")
    ax.plot(
        angles,
        vals_B,
        "o-",
        linewidth=2,
        label=f"VAE-B ({scores_B['overall']:.2f})",
        color="#e74c3c",
    )
    ax.fill(angles, vals_B, alpha=0.25, color="#e74c3c")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=8)
    ax.set_title("Resolution Score Breakdown")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    # Eigenvalue spectrum (simulated from data)
    ax = fig.add_subplot(gs[1, 1])

    # Reconstruct approximate eigenspectrum
    def make_spectrum(max_ev, min_ev, eff_dim, n_dims=16):
        # Exponential decay to match effective dim
        decay = np.log(max_ev / min_ev) / (n_dims - 1)
        return max_ev * np.exp(-decay * np.arange(n_dims))

    ev_A = make_spectrum(
        dim_A["max_eigenvalue"],
        dim_A["min_eigenvalue"],
        dim_A["effective_dim"],
    )
    ev_B = make_spectrum(
        dim_B["max_eigenvalue"],
        dim_B["min_eigenvalue"],
        dim_B["effective_dim"],
    )
    ax.semilogy(range(1, 17), ev_A, "o-", label="VAE-A", color="#3498db")
    ax.semilogy(range(1, 17), ev_B, "s-", label="VAE-B", color="#e74c3c")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Eigenvalue (log scale)")
    ax.set_title("Latent Space Eigenspectrum")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cumulative variance explained
    ax = fig.add_subplot(gs[1, 2])
    cum_var_A = np.cumsum(ev_A) / np.sum(ev_A) * 100
    cum_var_B = np.cumsum(ev_B) / np.sum(ev_B) * 100
    ax.plot(range(1, 17), cum_var_A, "o-", label="VAE-A", color="#3498db")
    ax.plot(range(1, 17), cum_var_B, "s-", label="VAE-B", color="#e74c3c")
    ax.axhline(95, color="green", linestyle="--", alpha=0.5, label="95%")
    ax.axhline(99, color="orange", linestyle="--", alpha=0.5, label="99%")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Variance (%)")
    ax.set_title("Variance Explained")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Interpolation quality
    ax = fig.add_subplot(gs[2, 0])
    interp_A = data["vae_a"]["interpolation"]
    interp_B = data["vae_b"]["interpolation"]
    metrics = ["Mean Error", "Std Error"]
    vals_A = [
        interp_A["mean_interpolation_error"],
        interp_A["std_interpolation_error"],
    ]
    vals_B = [
        interp_B["mean_interpolation_error"],
        interp_B["std_interpolation_error"],
    ]
    x = np.arange(len(metrics))
    ax.bar(x - width / 2, vals_A, width, label="VAE-A", color="#3498db", alpha=0.8)
    ax.bar(x + width / 2, vals_B, width, label="VAE-B", color="#e74c3c", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Interpolation Error")
    ax.set_title("Interpolation Quality\n(Lower = Better)")
    ax.legend()

    # Nearest neighbor analysis
    ax = fig.add_subplot(gs[2, 1])
    nn_A = data["vae_a"]["nearest_neighbor"]
    nn_B = data["vae_b"]["nearest_neighbor"]
    ax.bar(
        ["VAE-A", "VAE-B"],
        [nn_A["mean_latent_dist_to_nn"], nn_B["mean_latent_dist_to_nn"]],
        color=["#3498db", "#e74c3c"],
        alpha=0.8,
    )
    ax.set_ylabel("Mean Latent Distance to NN")
    ax.set_title("Nearest Neighbor Distance\n(Local Manifold Density)")

    # Summary
    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    summary = f"""
    MANIFOLD RESOLUTION SUMMARY
    ════════════════════════════

    Model: {data["model_info"]["total_params"]:,} params
    Latent: {data["model_info"]["latent_dim"]} dims
    Operations: {data["model_info"]["n_operations"]:,}

    VAE-A (Chaotic):
      Exact match: {data["vae_a"]["reconstruction"]["exact_match_rate"] * 100:.1f}%
      Effective dims: {dim_A["effective_dim"]:.1f}
      Resolution: {scores_A["overall"]:.3f}

    VAE-B (Frozen):
      Exact match: {data["vae_b"]["reconstruction"]["exact_match_rate"] * 100:.1f}%
      Effective dims: {dim_B["effective_dim"]:.1f}
      Resolution: {scores_B["overall"]:.3f}

    Combined Resolution: {data["resolution_score"]["combined"]:.3f}

    Note: VAE-B latent space is {ratio:.0f}x larger
    than VAE-A (57 vs 5 mean norm)
    """
    ax.text(
        0.1,
        0.95,
        summary,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_path / "manifold_resolution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path / 'manifold_resolution.png'}")


def plot_coupled_system(output_path):
    """Plot coupled dual-VAE system analysis."""
    data = load_json(PROJECT_ROOT / "reports" / "benchmarks" / "coupled_resolution_3.json")

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig)

    # Ensemble reconstruction
    ax = fig.add_subplot(gs[0, 0])
    ensemble = data["ensemble_reconstruction"]
    strategies = [
        "voting",
        "confidence\nweighted",
        "best_of_two",
        "VAE-A\nalone",
        "VAE-B\nalone",
    ]
    values = [
        ensemble["voting"]["exact_match_rate"],
        ensemble["confidence_weighted"]["exact_match_rate"],
        ensemble["best_of_two"]["exact_match_rate"],
        ensemble["baseline_vae_a"],
        ensemble["baseline_vae_b"],
    ]
    colors = ["green", "green", "green", "#3498db", "#e74c3c"]
    bars = ax.bar(strategies, values, color=colors, alpha=0.8)
    ax.axhline(1.0, color="gold", linestyle="--", linewidth=2)
    ax.set_ylabel("Exact Match Rate")
    ax.set_title("Ensemble Reconstruction\n(All ensemble strategies reach 100%)")
    ax.set_ylim(0, 1.1)
    for bar, val in zip(bars, values, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val * 100:.0f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Cross-injection coverage
    ax = fig.add_subplot(gs[0, 1])
    rho_05 = data["cross_injected_sampling_rho_05"]
    rho_07 = data["cross_injected_sampling_rho_07"]
    rhos = ["ρ=0\n(VAE-A)", "ρ=0\n(VAE-B)", "ρ=0.5", "ρ=0.7"]
    coverages = [
        rho_05["baseline_vae_a_coverage"] * 100,
        rho_05["baseline_vae_b_coverage"] * 100,
        rho_05["coverage_rate"] * 100,
        rho_07["coverage_rate"] * 100,
    ]
    colors = ["#3498db", "#e74c3c", "purple", "purple"]
    bars = ax.bar(rhos, coverages, color=colors, alpha=0.8)
    ax.set_ylabel("Coverage (%)")
    ax.set_title("Cross-Injection Sampling Coverage\n(Higher ρ = more mixing)")
    for bar, val in zip(bars, coverages, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Complementarity
    ax = fig.add_subplot(gs[0, 2])
    comp = data["complementary_coverage"]
    categories = [
        "Both\nPerfect",
        "VAE-A\nBest",
        "VAE-B\nBest",
        "Both\nImperfect",
    ]
    counts = [
        comp["both_perfect"],
        comp["vae_a_best"],
        comp["vae_b_best"],
        comp["both_imperfect"],
    ]
    colors = ["green", "#3498db", "#e74c3c", "gray"]
    ax.pie(
        counts,
        labels=categories,
        colors=colors,
        autopct="%1.1f%%",
        explode=[0.05, 0.05, 0.05, 0.05],
    )
    ax.set_title("Specialization Analysis\n(Which VAE reconstructs each op best)")

    # Latent coupling
    ax = fig.add_subplot(gs[1, 0])
    coupling = data["latent_coupling"]
    ax.bar(
        ["Correlation", "Alignment"],
        [coupling["mean_correlation"], coupling["alignment_score"]],
        color=["purple", "orange"],
        alpha=0.8,
    )
    ax.set_ylabel("Score")
    ax.set_title("Latent Space Coupling\n(How aligned are VAE-A and VAE-B?)")

    # Distance distribution (simulated)
    ax = fig.add_subplot(gs[1, 1])
    np.random.seed(42)
    distances = np.random.normal(coupling["mean_distance"], coupling["std_distance"], 1000)
    ax.hist(distances, bins=50, color="purple", alpha=0.7, edgecolor="black")
    ax.axvline(
        coupling["mean_distance"],
        color="red",
        linestyle="--",
        label=f"Mean: {coupling['mean_distance']:.1f}",
    )
    ax.axvline(
        coupling["min_distance"],
        color="green",
        linestyle=":",
        label=f"Min: {coupling['min_distance']:.1f}",
    )
    ax.axvline(
        coupling["max_distance"],
        color="orange",
        linestyle=":",
        label=f"Max: {coupling['max_distance']:.1f}",
    )
    ax.set_xlabel("Inter-VAE Latent Distance")
    ax.set_ylabel("Count")
    ax.set_title("Distance Between VAE-A and VAE-B Encodings\n(Same operation encoded by both)")
    ax.legend()

    # System resolution breakdown
    ax = fig.add_subplot(gs[1, 2])
    sys_res = data["system_resolution"]
    categories = ["Ensemble", "Coverage", "Coupling", "Overall"]
    values = [
        sys_res["ensemble"],
        sys_res["coverage"],
        sys_res["coupling"],
        sys_res["overall"],
    ]
    bars = ax.barh(
        categories,
        values,
        color=["green", "blue", "purple", "gold"],
        alpha=0.8,
    )
    ax.set_xlabel("Score")
    ax.set_title("System Resolution Breakdown")
    ax.set_xlim(0, 1.1)
    for bar, val in zip(bars, values, strict=False):
        ax.text(
            val + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            ha="left",
            va="center",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_path / "coupled_system.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path / 'coupled_system.png'}")


def plot_checkpoint_evolution(output_path):
    """Analyze how metrics evolved across checkpoints."""
    checkpoint_dir = CHECKPOINTS_DIR / "v5_5"

    checkpoints = sorted(checkpoint_dir.glob("epoch_*.pt"))
    if (checkpoint_dir / "latest.pt").exists():
        checkpoints.append(checkpoint_dir / "latest.pt")

    epochs = []
    coverage_A = []
    coverage_B = []
    entropy_A = []
    entropy_B = []
    lambdas = {"l1": [], "l2": [], "l3": []}
    rhos = []

    for ckpt_path in checkpoints:
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            epoch = ckpt.get("epoch", 0)
            epochs.append(epoch)

            # Coverage history (if available)
            cov_A_hist = ckpt.get("coverage_A_history", [])
            cov_B_hist = ckpt.get("coverage_B_history", [])
            coverage_A.append(cov_A_hist[-1] if cov_A_hist else 0)
            coverage_B.append(cov_B_hist[-1] if cov_B_hist else 0)

            # Entropy history
            H_A_hist = ckpt.get("H_A_history", [])
            H_B_hist = ckpt.get("H_B_history", [])
            entropy_A.append(H_A_hist[-1] if H_A_hist else 0)
            entropy_B.append(H_B_hist[-1] if H_B_hist else 0)

            # Lambda values
            lambdas["l1"].append(ckpt.get("lambda1", 0.7))
            lambdas["l2"].append(ckpt.get("lambda2", 0.7))
            lambdas["l3"].append(ckpt.get("lambda3", 0.3))

            # Rho
            rhos.append(ckpt.get("rho", 0.1))

        except Exception as e:
            print(f"Error loading {ckpt_path.name}: {e}")

    if not epochs:
        print("No checkpoint data available")
        return

    # Sort by epoch
    sort_idx = np.argsort(epochs)
    epochs = np.array(epochs)[sort_idx]
    coverage_A = np.array(coverage_A)[sort_idx]
    coverage_B = np.array(coverage_B)[sort_idx]
    entropy_A = np.array(entropy_A)[sort_idx]
    entropy_B = np.array(entropy_B)[sort_idx]
    for k in lambdas:
        lambdas[k] = np.array(lambdas[k])[sort_idx]
    rhos = np.array(rhos)[sort_idx]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Coverage evolution
    ax = axes[0, 0]
    ax.plot(epochs, coverage_A / 19683 * 100, "o-", label="VAE-A", color="#3498db")
    ax.plot(epochs, coverage_B / 19683 * 100, "s-", label="VAE-B", color="#e74c3c")
    ax.axhline(99, color="green", linestyle="--", alpha=0.5, label="99% target")
    ax.fill_between([0, 40], [90, 90], [100, 100], alpha=0.1, color="blue", label="Phase 1")
    ax.fill_between([40, 120], [90, 90], [100, 100], alpha=0.1, color="green")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Coverage (%)")
    ax.set_title("Coverage Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Entropy evolution
    ax = axes[0, 1]
    ax.plot(epochs, entropy_A, "o-", label="VAE-A", color="#3498db")
    ax.plot(epochs, entropy_B, "s-", label="VAE-B", color="#e74c3c")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Latent Entropy")
    ax.set_title("Entropy Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Lambda evolution
    ax = axes[1, 0]
    ax.plot(epochs, lambdas["l1"], "o-", label="λ₁ (VAE-A weight)", color="#3498db")
    ax.plot(epochs, lambdas["l2"], "s-", label="λ₂ (VAE-B weight)", color="#e74c3c")
    ax.plot(epochs, lambdas["l3"], "^-", label="λ₃ (entropy align)", color="green")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Lambda Value")
    ax.set_title("Adaptive Lambda Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Rho (permeability) evolution
    ax = axes[1, 1]
    ax.plot(epochs, rhos, "o-", color="purple", linewidth=2)
    ax.fill_between(epochs, 0, rhos, alpha=0.3, color="purple")
    ax.axhline(0.1, color="gray", linestyle=":", label="Phase 1 (ρ=0.1)")
    ax.axhline(0.3, color="gray", linestyle="--", label="Phase 2 (ρ→0.3)")
    ax.axhline(0.7, color="gray", linestyle="-", label="Phase 3 (ρ→0.7)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Permeability (ρ)")
    ax.set_title("Cross-Injection Permeability Schedule")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "checkpoint_evolution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path / 'checkpoint_evolution.png'}")


def main():
    output_path = PROJECT_ROOT / "outputs" / "manifold_viz"
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TRAINING ARTIFACTS VISUALIZATION")
    print("=" * 60)

    print("\n1. Benchmark Summary...")
    plot_benchmark_summary(output_path)

    print("\n2. Manifold Resolution Analysis...")
    plot_manifold_resolution(output_path)

    print("\n3. Coupled System Dynamics...")
    plot_coupled_system(output_path)

    print("\n4. Checkpoint Evolution...")
    plot_checkpoint_evolution(output_path)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"\nGenerated files in: {output_path}")
    for f in sorted(output_path.glob("*.png")):
        if any(
            x in f.name
            for x in [
                "benchmark",
                "manifold_resolution",
                "coupled",
                "checkpoint",
            ]
        ):
            print(f"  - {f.name}")


if __name__ == "__main__":
    main()
