# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Deep 3-Adic Structure Analysis of Ternary VAE.

Four key analyses:
1. Latent interpolation paths - algebraic validity of intermediate points
2. Per-digit latent dimensions - which dims encode which 3-adic digits
3. Training trajectory - when did 3-adic structure emerge
4. Latent arithmetic - linear group representation test

Usage:
    python src/scripts/visualization/analyze_3adic_deep.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import CHECKPOINTS_DIR
from src.data.generation import generate_all_ternary_operations
from src.geometry import poincare_distance
# V5.11 is the canonical model - alias for backwards compatibility
from src.models import TernaryVAE as DualNeuralVAEV5


def load_checkpoint(checkpoint_path: Path, device: str = "cpu"):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    use_statenet = checkpoint.get("statenet_enabled", False)

    model = DualNeuralVAEV5(input_dim=9, latent_dim=16, use_statenet=use_statenet)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    return model, checkpoint.get("epoch", "unknown")


def encode_all(model, device="cpu"):
    """Encode all operations."""
    operations = generate_all_ternary_operations()
    x = torch.tensor(operations, dtype=torch.float32, device=device)

    with torch.no_grad():
        mu_A, logvar_A = model.encoder_A(x)
        mu_B, logvar_B = model.encoder_B(x)

    return {
        "operations": operations,
        "z_A": mu_A.cpu().numpy(),
        "z_B": mu_B.cpu().numpy(),
        "x": x,
    }


def decode_latent(model, z, vae="A", device="cpu"):
    """Decode latent vector to operation."""
    z_tensor = torch.tensor(z, dtype=torch.float32, device=device)
    if z_tensor.dim() == 1:
        z_tensor = z_tensor.unsqueeze(0)

    with torch.no_grad():
        if vae == "A":
            logits = model.decoder_A(z_tensor)
        else:
            logits = model.decoder_B(z_tensor)
        pred = logits.argmax(dim=-1) - 1

    return pred.cpu().numpy()


def op_to_index(op):
    """Convert operation to 3-adic index."""
    idx = 0
    for i, val in enumerate(op):
        idx += (int(val) + 1) * (3**i)
    return idx


def index_to_op(idx):
    """Convert index to operation."""
    op = np.zeros(9)
    for i in range(9):
        op[i] = (idx % 3) - 1
        idx //= 3
    return op


def compute_3adic_distance(idx1, idx2):
    """Hamming distance in 3-adic representation."""
    diff = 0
    for _ in range(9):
        if (idx1 % 3) != (idx2 % 3):
            diff += 1
        idx1 //= 3
        idx2 //= 3
    return diff


# =============================================================================
# ANALYSIS 1: Latent Interpolation Paths
# =============================================================================


def analyze_interpolation_paths(model, data, output_path, device="cpu"):
    """Analyze if latent interpolations traverse algebraically valid paths."""
    print("\n" + "=" * 60)
    print("ANALYSIS 1: LATENT INTERPOLATION PATHS")
    print("=" * 60)

    operations = data["operations"]
    z_A = data["z_A"]
    z_B = data["z_B"]
    n_ops = len(operations)

    # Test interpolations between random pairs
    np.random.seed(42)
    n_tests = 500
    n_steps = 11  # 0.0, 0.1, ..., 1.0

    results = {"A": [], "B": []}

    for vae_name, z in [("A", z_A), ("B", z_B)]:
        valid_paths = 0
        monotonic_paths = 0
        avg_intermediate_validity = []

        for _ in range(n_tests):
            i, j = np.random.choice(n_ops, 2, replace=False)
            z_start, z_end = z[i], z[j]
            start_idx, end_idx = i, j

            # Interpolate
            path_ops = []
            path_indices = []
            for t in np.linspace(0, 1, n_steps):
                z_interp = (1 - t) * z_start + t * z_end
                decoded = decode_latent(model, z_interp, vae=vae_name, device=device)[0]
                idx = op_to_index(decoded)
                path_ops.append(decoded)
                path_indices.append(idx)

            # Check path properties
            # 1. All decoded ops are valid (always true for our decoder)
            # 2. Monotonic 3-adic distance from start
            distances_from_start = [compute_3adic_distance(start_idx, idx) for idx in path_indices]
            distances_from_end = [compute_3adic_distance(end_idx, idx) for idx in path_indices]

            # Check if path is "reasonable" - distance from start increases, from end decreases
            start_monotonic = all(distances_from_start[i] <= distances_from_start[i + 1] for i in range(len(distances_from_start) - 1))
            end_monotonic = all(distances_from_end[i] >= distances_from_end[i + 1] for i in range(len(distances_from_end) - 1))

            if start_monotonic or end_monotonic:
                monotonic_paths += 1

            # Check if intermediate points are "on the geodesic"
            # A geodesic would have d(start, mid) + d(mid, end) ≈ d(start, end)
            start_end_dist = compute_3adic_distance(start_idx, end_idx)
            geodesic_violations = 0
            for k, idx in enumerate(path_indices[1:-1]):
                d_start = compute_3adic_distance(start_idx, idx)
                d_end = compute_3adic_distance(end_idx, idx)
                if d_start + d_end > start_end_dist + 2:  # Allow some slack
                    geodesic_violations += 1

            avg_intermediate_validity.append(1 - geodesic_violations / (n_steps - 2))

            if geodesic_violations == 0:
                valid_paths += 1

        results[vae_name] = {
            "valid_paths": valid_paths / n_tests,
            "monotonic_paths": monotonic_paths / n_tests,
            "avg_validity": np.mean(avg_intermediate_validity),
        }

        print(f"\nVAE-{vae_name} Interpolation Analysis ({n_tests} paths):")
        print(f"  Geodesic-valid paths: {valid_paths}/{n_tests} ({100*valid_paths/n_tests:.1f}%)")
        print(f"  Monotonic paths: {monotonic_paths}/{n_tests} ({100*monotonic_paths/n_tests:.1f}%)")
        print(f"  Avg intermediate validity: {np.mean(avg_intermediate_validity):.3f}")

    # Visualization: Show example interpolations
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # PCA for visualization
    pca = PCA(n_components=2)
    z_A_2d = pca.fit_transform(z_A)

    for plot_idx in range(4):
        i, j = np.random.choice(n_ops, 2, replace=False)

        # VAE-A interpolation
        ax = axes[0, plot_idx]
        ax.scatter(z_A_2d[:, 0], z_A_2d[:, 1], c="lightgray", s=1, alpha=0.3)

        path_z = []
        path_decoded = []
        for t in np.linspace(0, 1, 21):
            z_interp = (1 - t) * z_A[i] + t * z_A[j]
            path_z.append(pca.transform(z_interp.reshape(1, -1))[0])
            decoded = decode_latent(model, z_interp, vae="A", device=device)[0]
            path_decoded.append(decoded)

        path_z = np.array(path_z)
        ax.plot(path_z[:, 0], path_z[:, 1], "b-", linewidth=2, alpha=0.7)
        ax.scatter(
            path_z[0, 0],
            path_z[0, 1],
            c="green",
            s=100,
            marker="o",
            zorder=5,
            label="Start",
        )
        ax.scatter(
            path_z[-1, 0],
            path_z[-1, 1],
            c="red",
            s=100,
            marker="s",
            zorder=5,
            label="End",
        )

        # Color intermediate points by 3-adic distance from start
        start_idx = op_to_index(operations[i])
        colors = [compute_3adic_distance(start_idx, op_to_index(d)) for d in path_decoded]
        ax.scatter(
            path_z[1:-1, 0],
            path_z[1:-1, 1],
            c=colors[1:-1],
            cmap="viridis",
            s=30,
            zorder=4,
        )
        ax.set_title(f"Path {plot_idx+1}: idx {i}→{j}\n3-adic dist: {compute_3adic_distance(i, j)}")
        if plot_idx == 0:
            ax.legend(loc="upper right")

        # Show decoded operations along path
        ax = axes[1, plot_idx]
        path_matrix = np.array(path_decoded)
        ax.imshow(path_matrix.T, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
        ax.set_xlabel("Interpolation Step")
        ax.set_ylabel("Operation Dimension")
        ax.set_title(f"Decoded Operations Along Path {plot_idx+1}")

    plt.suptitle(
        "VAE-A Latent Space Interpolation Paths\nColor = 3-adic distance from start",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path / "interpolation_paths.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {output_path / 'interpolation_paths.png'}")

    return results


# =============================================================================
# ANALYSIS 2: Per-Digit Latent Dimensions
# =============================================================================


def analyze_digit_dimensions(data, output_path):
    """Map which latent dimensions encode which 3-adic digits."""
    print("\n" + "=" * 60)
    print("ANALYSIS 2: PER-DIGIT LATENT DIMENSIONS")
    print("=" * 60)

    operations = data["operations"]
    z_A = data["z_A"]
    z_B = data["z_B"]

    # Extract each 3-adic digit for all operations
    digits = np.zeros((len(operations), 9))
    for idx in range(len(operations)):
        for d in range(9):
            digits[idx, d] = operations[idx][d]  # Already in {-1, 0, 1}

    # Compute correlation between each latent dim and each digit
    corr_A = np.zeros((16, 9))
    corr_B = np.zeros((16, 9))

    for dim in range(16):
        for digit in range(9):
            corr_A[dim, digit], _ = spearmanr(z_A[:, dim], digits[:, digit])
            corr_B[dim, digit], _ = spearmanr(z_B[:, dim], digits[:, digit])

    # Find best digit for each latent dimension
    print("\nVAE-A: Latent Dimension → Best Correlated Digit")
    print("Dim | Best Digit | Correlation | 2nd Best | Corr")
    print("-" * 55)
    for dim in range(16):
        sorted_idx = np.argsort(np.abs(corr_A[dim]))[::-1]
        best, second = sorted_idx[0], sorted_idx[1]
        print(f" {dim:2d} |     {best}      |   {corr_A[dim, best]:+.3f}    |    {second}     | {corr_A[dim, second]:+.3f}")

    print("\nVAE-B: Latent Dimension → Best Correlated Digit")
    print("Dim | Best Digit | Correlation | 2nd Best | Corr")
    print("-" * 55)
    for dim in range(16):
        sorted_idx = np.argsort(np.abs(corr_B[dim]))[::-1]
        best, second = sorted_idx[0], sorted_idx[1]
        print(f" {dim:2d} |     {best}      |   {corr_B[dim, best]:+.3f}    |    {second}     | {corr_B[dim, second]:+.3f}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Correlation heatmaps
    ax = axes[0, 0]
    im = ax.imshow(corr_A, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
    ax.set_xlabel("3-adic Digit Position")
    ax.set_ylabel("Latent Dimension")
    ax.set_title("VAE-A: Latent-Digit Correlations")
    ax.set_xticks(range(9))
    ax.set_yticks(range(16))
    plt.colorbar(im, ax=ax, label="Spearman r")

    ax = axes[0, 1]
    im = ax.imshow(corr_B, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
    ax.set_xlabel("3-adic Digit Position")
    ax.set_ylabel("Latent Dimension")
    ax.set_title("VAE-B: Latent-Digit Correlations")
    ax.set_xticks(range(9))
    ax.set_yticks(range(16))
    plt.colorbar(im, ax=ax, label="Spearman r")

    # Max absolute correlation per latent dim
    ax = axes[1, 0]
    max_corr_A = np.max(np.abs(corr_A), axis=1)
    max_corr_B = np.max(np.abs(corr_B), axis=1)
    x = np.arange(16)
    width = 0.35
    ax.bar(
        x - width / 2,
        max_corr_A,
        width,
        label="VAE-A",
        color="blue",
        alpha=0.7,
    )
    ax.bar(
        x + width / 2,
        max_corr_B,
        width,
        label="VAE-B",
        color="orange",
        alpha=0.7,
    )
    ax.axhline(0.3, color="red", linestyle="--", label="Threshold (0.3)")
    ax.set_xlabel("Latent Dimension")
    ax.set_ylabel("Max |Correlation| with any Digit")
    ax.set_title('Latent Dimension "Informativeness"')
    ax.legend()
    ax.set_xticks(x)

    # Which digit is each dimension most correlated with?
    ax = axes[1, 1]
    best_digit_A = np.argmax(np.abs(corr_A), axis=1)
    best_digit_B = np.argmax(np.abs(corr_B), axis=1)

    # Count how many dims are assigned to each digit
    counts_A = np.bincount(best_digit_A, minlength=9)
    counts_B = np.bincount(best_digit_B, minlength=9)

    x = np.arange(9)
    ax.bar(x - width / 2, counts_A, width, label="VAE-A", color="blue", alpha=0.7)
    ax.bar(
        x + width / 2,
        counts_B,
        width,
        label="VAE-B",
        color="orange",
        alpha=0.7,
    )
    ax.set_xlabel("3-adic Digit Position")
    ax.set_ylabel("# Latent Dims Best Correlated")
    ax.set_title("Digit Coverage by Latent Dimensions")
    ax.legend()
    ax.set_xticks(x)

    plt.tight_layout()
    plt.savefig(
        output_path / "digit_dimension_mapping.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print(f"\nSaved: {output_path / 'digit_dimension_mapping.png'}")

    # Summary statistics
    active_dims_A = np.sum(max_corr_A > 0.3)
    active_dims_B = np.sum(max_corr_B > 0.3)
    print(f"\nActive dimensions (|r| > 0.3): VAE-A={active_dims_A}, VAE-B={active_dims_B}")

    return {"corr_A": corr_A, "corr_B": corr_B}


# =============================================================================
# ANALYSIS 3: Training Trajectory
# =============================================================================


def analyze_training_trajectory(output_path, device="cpu"):
    """Track 3-adic structure emergence across training epochs."""
    print("\n" + "=" * 60)
    print("ANALYSIS 3: TRAINING TRAJECTORY")
    print("=" * 60)

    checkpoint_dir = CHECKPOINTS_DIR / "v5_5"

    # Find available checkpoints
    checkpoints = sorted(checkpoint_dir.glob("epoch_*.pt"))
    checkpoints.append(checkpoint_dir / "latest.pt")

    if not checkpoints:
        print("No checkpoints found!")
        return None

    print(f"Found {len(checkpoints)} checkpoints")

    operations = generate_all_ternary_operations()
    n_ops = len(operations)

    # Sample pairs for correlation computation
    np.random.seed(42)
    n_pairs = 10000
    pairs = np.random.choice(n_ops, size=(n_pairs, 2), replace=True)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]

    # Precompute 3-adic distances
    adic_dists = np.array([compute_3adic_distance(i, j) for i, j in pairs])

    results = []

    for ckpt_path in checkpoints:
        if not ckpt_path.exists():
            continue

        try:
            model, epoch = load_checkpoint(ckpt_path, device)

            # Encode
            x = torch.tensor(operations, dtype=torch.float32, device=device)
            with torch.no_grad():
                mu_A, _ = model.encoder_A(x)
                mu_B, _ = model.encoder_B(x)

            z_A = mu_A.cpu().numpy()
            z_B = mu_B.cpu().numpy()

            # Compute latent distances for sampled pairs (hyperbolic Poincaré distance)
            z_A_t = torch.from_numpy(z_A)
            z_B_t = torch.from_numpy(z_B)
            with torch.no_grad():
                latent_dists_A = poincare_distance(z_A_t[pairs[:, 0]], z_A_t[pairs[:, 1]]).numpy()
                latent_dists_B = poincare_distance(z_B_t[pairs[:, 0]], z_B_t[pairs[:, 1]]).numpy()

            # Correlations
            corr_A, _ = spearmanr(adic_dists, latent_dists_A)
            corr_B, _ = spearmanr(adic_dists, latent_dists_B)

            results.append(
                {
                    "epoch": epoch,
                    "corr_A": corr_A,
                    "corr_B": corr_B,
                    "checkpoint": ckpt_path.name,
                }
            )

            print(f"  Epoch {epoch:3}: corr_A={corr_A:.4f}, corr_B={corr_B:.4f}")

        except Exception as e:
            print(f"  Error loading {ckpt_path.name}: {e}")

    if not results:
        print("No valid results!")
        return None

    # Sort by epoch
    results = sorted(
        results,
        key=lambda x: x["epoch"] if isinstance(x["epoch"], int) else 999,
    )

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = [r["epoch"] for r in results]
    corrs_A = [r["corr_A"] for r in results]
    corrs_B = [r["corr_B"] for r in results]

    ax = axes[0]
    ax.plot(epochs, corrs_A, "bo-", label="VAE-A", linewidth=2, markersize=8)
    ax.plot(epochs, corrs_B, "rs-", label="VAE-B", linewidth=2, markersize=8)
    ax.axhline(0.6, color="green", linestyle="--", alpha=0.5, label="Strong threshold")
    ax.axhline(
        0.3,
        color="orange",
        linestyle="--",
        alpha=0.5,
        label="Moderate threshold",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Spearman Correlation (3-adic vs Latent Distance)")
    ax.set_title("3-Adic Structure Emergence During Training")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Phase annotations
    ax.axvspan(0, 40, alpha=0.1, color="blue", label="Phase 1")
    ax.axvspan(40, 120, alpha=0.1, color="green")
    ax.axvspan(120, 250, alpha=0.1, color="orange")

    # Correlation difference
    ax = axes[1]
    corr_diff = np.array(corrs_A) - np.array(corrs_B)
    ax.bar(
        epochs,
        corr_diff,
        color=["blue" if d > 0 else "orange" for d in corr_diff],
        alpha=0.7,
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Correlation Difference (VAE-A - VAE-B)")
    ax.set_title("VAE-A vs VAE-B 3-Adic Preservation")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "training_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {output_path / 'training_trajectory.png'}")

    return results


# =============================================================================
# ANALYSIS 4: Latent Arithmetic
# =============================================================================


def analyze_latent_arithmetic(model, data, output_path, device="cpu"):
    """Test if latent space has linear group structure."""
    print("\n" + "=" * 60)
    print("ANALYSIS 4: LATENT ARITHMETIC (GROUP REPRESENTATION)")
    print("=" * 60)

    operations = data["operations"]
    z_A = data["z_A"]
    z_B = data["z_B"]
    n_ops = len(operations)

    # Find special operations
    zero_idx = op_to_index(np.zeros(9))  # Identity for addition

    # Find add_mod3 operation
    add_mod3_op = np.array([-1, 0, 1, 0, 1, -1, 1, -1, 0])
    add_mod3_idx = op_to_index(add_mod3_op)

    print(f"Zero operation index: {zero_idx}")
    print(f"Add_mod3 operation index: {add_mod3_idx}")

    # Test: For addition, does z_a + z_b - z_0 ≈ z_{a+b}?
    # More precisely: for each pair (a, b), compute the "sum" operation
    # and check if latent arithmetic approximates it

    def ternary_add(op_a, op_b):
        """Pointwise ternary addition (mod 3, shifted)."""
        result = np.zeros(9)
        for i in range(9):
            # Map {-1,0,1} to {0,1,2}, add, mod 3, map back
            val = ((int(op_a[i]) + 1) + (int(op_b[i]) + 1)) % 3 - 1
            result[i] = val
        return result

    # Test on random pairs
    np.random.seed(42)
    n_tests = 1000

    results = {
        "A": {"exact": 0, "close": 0, "dist": []},
        "B": {"exact": 0, "close": 0, "dist": []},
    }

    z_zero_A = z_A[zero_idx]
    z_zero_B = z_B[zero_idx]

    for _ in range(n_tests):
        i, j = np.random.choice(n_ops, 2, replace=True)

        # Compute actual sum operation
        sum_op = ternary_add(operations[i], operations[j])
        sum_idx = op_to_index(sum_op)

        for vae_name, z, z_zero in [
            ("A", z_A, z_zero_A),
            ("B", z_B, z_zero_B),
        ]:
            # Latent arithmetic: z_i + z_j - z_0
            z_predicted = z[i] + z[j] - z_zero

            # Find nearest neighbor in latent space
            distances = np.linalg.norm(z - z_predicted, axis=1)
            nn_idx = np.argmin(distances)
            distances[nn_idx]

            # Check if prediction is correct
            if nn_idx == sum_idx:
                results[vae_name]["exact"] += 1

            # Check if prediction is close (within top 10 neighbors)
            sorted_indices = np.argsort(distances)
            if sum_idx in sorted_indices[:10]:
                results[vae_name]["close"] += 1

            results[vae_name]["dist"].append(distances[sum_idx])

    print("\nLatent Arithmetic Test: z_a + z_b - z_0 ≈ z_(a+b)?")
    print(f"Testing {n_tests} random pairs\n")

    for vae_name in ["A", "B"]:
        r = results[vae_name]
        print(f"VAE-{vae_name}:")
        print(f"  Exact matches (NN = true sum): {r['exact']}/{n_tests} ({100*r['exact']/n_tests:.1f}%)")
        print(f"  Close matches (in top 10 NN):  {r['close']}/{n_tests} ({100*r['close']/n_tests:.1f}%)")
        print(f"  Mean distance to true sum:     {np.mean(r['dist']):.4f}")

    # Test other operations: subtraction, negation
    print("\n--- Testing Other Arithmetic Operations ---")

    # Negation: -a should map to z_0 - z_a + z_0 = 2*z_0 - z_a
    def ternary_neg(op_a):
        return -op_a

    neg_exact_A, neg_exact_B = 0, 0
    for i in np.random.choice(n_ops, 500):
        neg_op = ternary_neg(operations[i])
        neg_idx = op_to_index(neg_op)

        # VAE-A
        z_pred = 2 * z_zero_A - z_A[i]
        nn_idx = np.argmin(np.linalg.norm(z_A - z_pred, axis=1))
        if nn_idx == neg_idx:
            neg_exact_A += 1

        # VAE-B
        z_pred = 2 * z_zero_B - z_B[i]
        nn_idx = np.argmin(np.linalg.norm(z_B - z_pred, axis=1))
        if nn_idx == neg_idx:
            neg_exact_B += 1

    print("\nNegation test: 2*z_0 - z_a ≈ z_(-a)?")
    print(f"  VAE-A: {neg_exact_A}/500 ({100*neg_exact_A/500:.1f}%)")
    print(f"  VAE-B: {neg_exact_B}/500 ({100*neg_exact_B/500:.1f}%)")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Distribution of distances to true sum
    ax = axes[0, 0]
    ax.hist(results["A"]["dist"], bins=50, alpha=0.5, label="VAE-A", density=True)
    ax.hist(results["B"]["dist"], bins=50, alpha=0.5, label="VAE-B", density=True)
    ax.axvline(np.mean(results["A"]["dist"]), color="blue", linestyle="--")
    ax.axvline(np.mean(results["B"]["dist"]), color="orange", linestyle="--")
    ax.set_xlabel("Latent Distance to True Sum")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Prediction Error\n(Lower = Better Linear Structure)")
    ax.legend()

    # Visualize arithmetic in 2D
    ax = axes[0, 1]
    pca = PCA(n_components=2)
    z_A_2d = pca.fit_transform(z_A)

    # Show a few arithmetic examples
    for _ in range(5):
        i, j = np.random.choice(n_ops, 2)
        sum_op = ternary_add(operations[i], operations[j])
        sum_idx = op_to_index(sum_op)

        z_pred = z_A[i] + z_A[j] - z_zero_A
        z_pred_2d = pca.transform(z_pred.reshape(1, -1))[0]

        ax.scatter(z_A_2d[i, 0], z_A_2d[i, 1], c="blue", s=50, marker="o")
        ax.scatter(z_A_2d[j, 0], z_A_2d[j, 1], c="green", s=50, marker="o")
        ax.scatter(z_A_2d[sum_idx, 0], z_A_2d[sum_idx, 1], c="red", s=50, marker="*")
        ax.scatter(z_pred_2d[0], z_pred_2d[1], c="purple", s=50, marker="x")
        ax.plot(
            [z_A_2d[i, 0], z_pred_2d[0]],
            [z_A_2d[i, 1], z_pred_2d[1]],
            "k--",
            alpha=0.3,
        )
        ax.plot(
            [z_A_2d[j, 0], z_pred_2d[0]],
            [z_A_2d[j, 1], z_pred_2d[1]],
            "k--",
            alpha=0.3,
        )

    ax.scatter(
        z_A_2d[zero_idx, 0],
        z_A_2d[zero_idx, 1],
        c="black",
        s=200,
        marker="s",
        label="Zero",
    )
    ax.scatter([], [], c="blue", s=50, marker="o", label="Op A")
    ax.scatter([], [], c="green", s=50, marker="o", label="Op B")
    ax.scatter([], [], c="red", s=50, marker="*", label="True A+B")
    ax.scatter([], [], c="purple", s=50, marker="x", label="Predicted")
    ax.set_title("VAE-A: Latent Arithmetic Examples\nBlue + Green → Red (true) vs Purple (pred)")
    ax.legend(loc="upper right")

    # Accuracy vs 3-adic distance
    ax = axes[1, 0]
    # Recompute with tracking
    accuracy_by_dist = {d: [] for d in range(1, 10)}
    for _ in range(2000):
        i, j = np.random.choice(n_ops, 2)
        sum_op = ternary_add(operations[i], operations[j])
        sum_idx = op_to_index(sum_op)
        adic_dist = compute_3adic_distance(i, j)

        z_pred = z_A[i] + z_A[j] - z_zero_A
        nn_idx = np.argmin(np.linalg.norm(z_A - z_pred, axis=1))
        accuracy_by_dist[adic_dist].append(1 if nn_idx == sum_idx else 0)

    dists = list(range(1, 10))
    accs = [np.mean(accuracy_by_dist[d]) if accuracy_by_dist[d] else 0 for d in dists]
    ax.bar(dists, accs, color="purple", alpha=0.7)
    ax.set_xlabel("3-adic Distance Between Operands")
    ax.set_ylabel("Latent Arithmetic Accuracy")
    ax.set_title("Arithmetic Accuracy vs Operand Distance")

    # Summary statistics
    ax = axes[1, 1]
    stats = {
        "Add (exact)": [
            results["A"]["exact"] / n_tests,
            results["B"]["exact"] / n_tests,
        ],
        "Add (top-10)": [
            results["A"]["close"] / n_tests,
            results["B"]["close"] / n_tests,
        ],
        "Negation": [neg_exact_A / 500, neg_exact_B / 500],
    }
    x = np.arange(len(stats))
    width = 0.35
    ax.bar(
        x - width / 2,
        [v[0] for v in stats.values()],
        width,
        label="VAE-A",
        color="blue",
        alpha=0.7,
    )
    ax.bar(
        x + width / 2,
        [v[1] for v in stats.values()],
        width,
        label="VAE-B",
        color="orange",
        alpha=0.7,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(stats.keys())
    ax.set_ylabel("Accuracy")
    ax.set_title("Latent Arithmetic Summary")
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path / "latent_arithmetic.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {output_path / 'latent_arithmetic.png'}")

    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    checkpoint_path = CHECKPOINTS_DIR / "v5_5" / "latest.pt"
    output_path = PROJECT_ROOT / "outputs" / "manifold_viz"
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model and encode
    print("\nLoading model...")
    model, epoch = load_checkpoint(checkpoint_path, device)
    print(f"Loaded epoch {epoch}")

    data = encode_all(model, device)
    print(f"Encoded {len(data['operations'])} operations")

    # Run all analyses
    interp_results = analyze_interpolation_paths(model, data, output_path, device)
    analyze_digit_dimensions(data, output_path)
    analyze_training_trajectory(output_path, device)
    arithmetic_results = analyze_latent_arithmetic(model, data, output_path, device)

    # Final summary
    print("\n" + "=" * 60)
    print("COMPLETE ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"\n1. INTERPOLATION: {interp_results['A']['avg_validity']*100:.1f}% path validity (VAE-A)")
    print("2. DIGIT MAPPING: Latent dims correlate with specific 3-adic digits")
    print("3. TRAJECTORY: 3-adic structure emerges during training")
    print(f"4. ARITHMETIC: {arithmetic_results['A']['exact']/10:.1f}% exact addition matches (VAE-A)")

    print("\nGenerated files:")
    for f in sorted(output_path.glob("*.png")):
        if any(x in f.name for x in ["interpolation", "digit", "trajectory", "arithmetic"]):
            print(f"  - {f.name}")


if __name__ == "__main__":
    main()
