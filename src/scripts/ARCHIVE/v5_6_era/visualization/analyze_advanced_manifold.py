#!/usr/bin/env python3
# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""
Advanced Manifold Analysis for Ternary VAE

Implements 4 deep visualizations:
1. Jacobian norm surface - decoder sensitivity/stability
2. Voronoi cells - decision boundaries in latent space
3. Latent holes - gaps in manifold coverage
4. Per-operation difficulty ranking - algebraic outliers

Usage:
    python src/scripts/visualization/analyze_advanced_manifold.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy.spatial import Voronoi, voronoi_plot_2d

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.ternary_vae_v5_6 import DualNeuralVAEV5

from src.config.paths import CHECKPOINTS_DIR


def load_checkpoint(checkpoint_path: str, device: str = "cuda"):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract config
    config = checkpoint.get("config", {"input_dim": 9, "latent_dim": 16})

    # Create model
    model = DualNeuralVAEV5(
        input_dim=config.get("input_dim", 9),
        latent_dim=config.get("latent_dim", 16),
    ).to(device)

    # Load state - handle different checkpoint formats
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model, config


def generate_all_operations(device: str = "cuda"):
    """Generate all 19683 ternary operations as one-hot tensors."""
    num_ops = 3**9
    operations = torch.zeros(num_ops, 8, 3, device=device)

    for idx in range(num_ops):
        digits = []
        temp = idx
        for _ in range(9):
            digits.append(temp % 3)
            temp //= 3
        # First 8 digits define the operation
        for pos, digit in enumerate(digits[:8]):
            operations[idx, pos, digit] = 1.0

    return operations


def encode_all_operations(model, operations, batch_size: int = 1024):
    """Encode all operations to latent space."""
    model.eval()
    all_z = []
    all_mu = []
    all_logvar = []

    with torch.no_grad():
        for i in range(0, len(operations), batch_size):
            batch = operations[i : i + batch_size]
            # Flatten one-hot to input format: [batch, 8, 3] -> [batch, 8*3] -> use argmax [batch, 8]
            batch_flat = batch.argmax(dim=-1).float()  # [batch, 8]
            # Pad to 9 dimensions (input_dim=9)
            batch_padded = F.pad(batch_flat, (0, 1), value=0)  # [batch, 9]

            # Use encoder_A for encoding
            mu, logvar = model.encoder_A(batch_padded)
            # In eval mode, z = mu
            z = model.reparameterize(mu, logvar)

            all_z.append(z.cpu())
            all_mu.append(mu.cpu())
            all_logvar.append(logvar.cpu())

    return (
        torch.cat(all_z, dim=0),
        torch.cat(all_mu, dim=0),
        torch.cat(all_logvar, dim=0),
    )


def compute_jacobian_surface(model, latent_codes, device="cuda", grid_size=50):
    """
    Compute Jacobian norm surface showing decoder sensitivity.

    The Jacobian ||∂f/∂z|| measures how much the decoder output changes
    for small perturbations in latent space. High values = sensitive regions.
    """
    print("Computing Jacobian norm surface...")

    # Get 2D projection via PCA
    z_np = latent_codes.numpy()
    z_centered = z_np - z_np.mean(axis=0)
    U, S, Vt = np.linalg.svd(z_centered, full_matrices=False)
    z_2d = z_centered @ Vt[:2].T

    # Create grid over 2D projection
    x_min, x_max = z_2d[:, 0].min() - 0.5, z_2d[:, 0].max() + 0.5
    y_min, y_max = z_2d[:, 1].min() - 0.5, z_2d[:, 1].max() + 0.5

    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Compute Jacobian norm at each grid point
    jacobian_norms = np.zeros((grid_size, grid_size))

    model.eval()
    mean_z = z_np.mean(axis=0)

    for i in range(grid_size):
        for j in range(grid_size):
            # Project 2D grid point back to full latent space
            # Use the two principal components, rest at mean
            z_point = mean_z.copy()
            z_point = z_point + X[i, j] * Vt[0] + Y[i, j] * Vt[1]
            z_tensor = torch.tensor(z_point, dtype=torch.float32, device=device).unsqueeze(0)
            z_tensor.requires_grad_(True)

            # Forward through decoder
            output = model.decoder_A(z_tensor)

            # Compute Frobenius norm of Jacobian via backprop
            # ||J||_F = sqrt(sum of squared partial derivatives)
            jacobian_norm = 0.0
            output_flat = output.view(-1)

            for k in range(min(24, len(output_flat))):  # Sample outputs for speed
                if z_tensor.grad is not None:
                    z_tensor.grad.zero_()
                output_flat[k].backward(retain_graph=True)
                if z_tensor.grad is not None:
                    jacobian_norm += z_tensor.grad.norm().item() ** 2

            jacobian_norms[i, j] = np.sqrt(jacobian_norm)

    # Smooth for visualization
    jacobian_norms = gaussian_filter(jacobian_norms, sigma=1.0)

    return X, Y, jacobian_norms, z_2d, Vt


def compute_voronoi_cells(latent_codes, n_display=200):
    """
    Compute Voronoi diagram showing decision boundaries.

    Each cell represents the region of latent space that maps to a specific operation.
    """
    print("Computing Voronoi cells...")

    # Get 2D projection
    z_np = latent_codes.numpy()
    z_centered = z_np - z_np.mean(axis=0)
    U, S, Vt = np.linalg.svd(z_centered, full_matrices=False)
    z_2d = z_centered @ Vt[:2].T

    # Subsample for Voronoi (too many points makes it unreadable)
    indices = np.random.choice(len(z_2d), min(n_display, len(z_2d)), replace=False)
    points = z_2d[indices]

    # Add boundary points to bound the diagram
    x_range = z_2d[:, 0].max() - z_2d[:, 0].min()
    y_range = z_2d[:, 1].max() - z_2d[:, 1].min()
    boundary = np.array(
        [
            [z_2d[:, 0].min() - x_range, z_2d[:, 1].min() - y_range],
            [z_2d[:, 0].max() + x_range, z_2d[:, 1].min() - y_range],
            [z_2d[:, 0].min() - x_range, z_2d[:, 1].max() + y_range],
            [z_2d[:, 0].max() + x_range, z_2d[:, 1].max() + y_range],
        ]
    )
    points_with_boundary = np.vstack([points, boundary])

    vor = Voronoi(points_with_boundary)

    return vor, z_2d, indices, points


def detect_latent_holes(latent_codes, grid_size=100, threshold_percentile=10):
    """
    Detect holes/gaps in the latent manifold.

    Uses density estimation to find regions with few or no encoded operations.
    """
    print("Detecting latent holes...")

    # Get 2D projection
    z_np = latent_codes.numpy()
    z_centered = z_np - z_np.mean(axis=0)
    U, S, Vt = np.linalg.svd(z_centered, full_matrices=False)
    z_2d = z_centered @ Vt[:2].T

    # Create density grid
    x_min, x_max = z_2d[:, 0].min() - 0.5, z_2d[:, 0].max() + 0.5
    y_min, y_max = z_2d[:, 1].min() - 0.5, z_2d[:, 1].max() + 0.5

    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Compute density via histogram
    density, x_edges, y_edges = np.histogram2d(
        z_2d[:, 0],
        z_2d[:, 1],
        bins=grid_size,
        range=[[x_min, x_max], [y_min, y_max]],
    )

    # Smooth density
    density_smooth = gaussian_filter(density, sigma=2.0)

    # Find holes (low density regions within the convex hull)
    from scipy.spatial import ConvexHull, Delaunay

    hull = ConvexHull(z_2d)
    hull_delaunay = Delaunay(z_2d[hull.vertices])

    # Create mask for points inside hull
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    inside_hull = hull_delaunay.find_simplex(grid_points) >= 0
    inside_mask = inside_hull.reshape(grid_size, grid_size)

    # Identify holes: low density but inside hull
    threshold = np.percentile(density_smooth[density_smooth > 0], threshold_percentile)
    holes = (threshold > density_smooth.T) & inside_mask

    return X, Y, density_smooth.T, holes, z_2d


def compute_operation_difficulty(model, operations, batch_size=1024, device="cuda"):
    """
    Rank operations by reconstruction difficulty.

    Returns per-operation reconstruction loss to identify algebraic outliers.
    """
    print("Computing per-operation difficulty...")

    model.eval()
    difficulties = []

    with torch.no_grad():
        for i in range(0, len(operations), batch_size):
            batch = operations[i : i + batch_size]
            # Convert to model input format
            batch_flat = batch.argmax(dim=-1).float()  # [batch, 8]
            batch_padded = F.pad(batch_flat, (0, 1), value=0)  # [batch, 9]

            # Encode
            mu, logvar = model.encoder_A(batch_padded)
            z = model.reparameterize(mu, logvar)

            # Decode
            recon = model.decoder_A(z)  # [batch, 9, 3]

            # Per-sample reconstruction loss (only first 8 positions matter)
            recon_8 = recon[:, :8, :]  # [batch, 8, 3]
            recon_flat = recon_8.reshape(-1, 3)
            target_flat = batch.argmax(dim=-1).view(-1)

            ce_loss = F.cross_entropy(recon_flat, target_flat, reduction="none")
            ce_loss = ce_loss.view(batch.size(0), -1).mean(dim=1)

            # KL divergence per sample
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

            # Total difficulty
            total_loss = ce_loss + 0.001 * kl_loss
            difficulties.extend(total_loss.cpu().numpy())

    return np.array(difficulties)


def get_operation_string(idx):
    """Convert operation index to digit string."""
    digits = []
    temp = idx
    for _ in range(9):
        digits.append(str(temp % 3))
        temp //= 3
    return "".join(digits[:8])


def plot_all_analyses(model, operations, latent_codes, device="cuda"):
    """Generate all 4 advanced visualizations."""

    output_dir = project_root / "outputs" / "manifold_viz"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Jacobian Norm Surface
    print("\n=== 1. Jacobian Norm Surface ===")
    X_j, Y_j, jacobian_norms, z_2d_j, Vt_j = compute_jacobian_surface(model, latent_codes, device=device, grid_size=40)

    fig = plt.figure(figsize=(14, 5))

    # 3D surface
    ax1 = fig.add_subplot(121, projection="3d")
    surf = ax1.plot_surface(X_j, Y_j, jacobian_norms, cmap="magma", alpha=0.8)
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_zlabel("||∂f/∂z||")
    ax1.set_title("Decoder Jacobian Norm Surface\n(Higher = More Sensitive)")
    fig.colorbar(surf, ax=ax1, shrink=0.5, label="Jacobian Norm")

    # 2D heatmap with points
    ax2 = fig.add_subplot(122)
    im = ax2.contourf(X_j, Y_j, jacobian_norms, levels=20, cmap="magma")
    ax2.scatter(z_2d_j[:, 0], z_2d_j[:, 1], c="white", s=1, alpha=0.3)
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_title("Jacobian Sensitivity Map\n(Operations shown in white)")
    fig.colorbar(im, ax=ax2, label="Jacobian Norm")

    plt.tight_layout()
    plt.savefig(output_dir / "jacobian_surface.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Jacobian norm range: [{jacobian_norms.min():.3f}, {jacobian_norms.max():.3f}]")
    print(f"  Mean sensitivity: {jacobian_norms.mean():.3f}")
    print("  Saved: jacobian_surface.png")

    # 2. Voronoi Cells
    print("\n=== 2. Voronoi Cells ===")
    vor, z_2d_v, indices, points = compute_voronoi_cells(latent_codes, n_display=150)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Full Voronoi diagram
    ax1 = axes[0]
    voronoi_plot_2d(
        vor,
        ax=ax1,
        show_vertices=False,
        line_colors="blue",
        line_width=0.5,
        line_alpha=0.6,
        point_size=2,
    )
    ax1.scatter(z_2d_v[:, 0], z_2d_v[:, 1], c="gray", s=1, alpha=0.2, label="All ops")
    ax1.scatter(points[:, 0], points[:, 1], c="red", s=10, alpha=0.8, label="Sampled")
    ax1.set_xlim(z_2d_v[:, 0].min() - 0.5, z_2d_v[:, 0].max() + 0.5)
    ax1.set_ylim(z_2d_v[:, 1].min() - 0.5, z_2d_v[:, 1].max() + 0.5)
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_title("Voronoi Decision Boundaries\n(Each cell → one operation)")
    ax1.legend(loc="upper right")

    # Zoomed region
    ax2 = axes[1]
    center_x, center_y = z_2d_v[:, 0].mean(), z_2d_v[:, 1].mean()
    zoom_range = 1.0
    voronoi_plot_2d(
        vor,
        ax=ax2,
        show_vertices=False,
        line_colors="blue",
        line_width=1.0,
        line_alpha=0.8,
        point_size=5,
    )
    ax2.scatter(points[:, 0], points[:, 1], c="red", s=30, alpha=0.9)
    ax2.set_xlim(center_x - zoom_range, center_x + zoom_range)
    ax2.set_ylim(center_y - zoom_range, center_y + zoom_range)
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_title("Zoomed Voronoi Cells\n(Center region)")

    plt.tight_layout()
    plt.savefig(output_dir / "voronoi_cells.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Compute cell size statistics
    cell_areas = []
    for region_idx in vor.point_region[: len(points)]:
        region = vor.regions[region_idx]
        if -1 not in region and len(region) > 0:
            polygon = vor.vertices[region]
            # Shoelace formula for area
            n = len(polygon)
            area = 0.5 * abs(
                sum(polygon[i, 0] * polygon[(i + 1) % n, 1] - polygon[(i + 1) % n, 0] * polygon[i, 1] for i in range(n))
            )
            cell_areas.append(area)

    print(f"  Voronoi cells computed: {len(points)}")
    if cell_areas:
        print(f"  Cell area range: [{min(cell_areas):.4f}, {max(cell_areas):.4f}]")
        print(f"  Mean cell area: {np.mean(cell_areas):.4f}")
    print("  Saved: voronoi_cells.png")

    # 3. Latent Holes
    print("\n=== 3. Latent Holes Detection ===")
    X_h, Y_h, density, holes, z_2d_h = detect_latent_holes(latent_codes, grid_size=80)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Density map
    ax1 = axes[0]
    im1 = ax1.contourf(X_h, Y_h, density, levels=30, cmap="viridis")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_title("Latent Density Map")
    fig.colorbar(im1, ax=ax1, label="Point Density")

    # Holes overlay
    ax2 = axes[1]
    ax2.contourf(X_h, Y_h, density, levels=30, cmap="viridis", alpha=0.5)
    ax2.contour(X_h, Y_h, holes.astype(float), levels=[0.5], colors="red", linewidths=2)
    ax2.scatter(z_2d_h[:, 0], z_2d_h[:, 1], c="white", s=1, alpha=0.3)
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_title("Manifold Holes\n(Red = gaps in coverage)")

    # Hole regions only
    ax3 = axes[2]
    ax3.imshow(
        holes.astype(float),
        extent=[X_h.min(), X_h.max(), Y_h.min(), Y_h.max()],
        origin="lower",
        cmap="Reds",
        aspect="auto",
    )
    ax3.scatter(z_2d_h[:, 0], z_2d_h[:, 1], c="blue", s=1, alpha=0.2)
    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")
    ax3.set_title("Hole Regions\n(Red = low density)")

    plt.tight_layout()
    plt.savefig(output_dir / "latent_holes.png", dpi=150, bbox_inches="tight")
    plt.close()

    hole_fraction = holes.sum() / (holes.shape[0] * holes.shape[1]) * 100
    print(f"  Hole coverage: {hole_fraction:.2f}% of latent area")
    print(f"  Density range: [{density.min():.1f}, {density.max():.1f}]")
    print("  Saved: latent_holes.png")

    # 4. Per-Operation Difficulty
    print("\n=== 4. Per-Operation Difficulty Ranking ===")
    difficulties = compute_operation_difficulty(model, operations, device=device)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Difficulty histogram
    ax1 = axes[0, 0]
    ax1.hist(difficulties, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    ax1.axvline(
        np.percentile(difficulties, 95),
        color="red",
        linestyle="--",
        label=f"95th percentile: {np.percentile(difficulties, 95):.4f}",
    )
    ax1.axvline(
        np.percentile(difficulties, 99),
        color="darkred",
        linestyle="--",
        label=f"99th percentile: {np.percentile(difficulties, 99):.4f}",
    )
    ax1.set_xlabel("Reconstruction Difficulty")
    ax1.set_ylabel("Count")
    ax1.set_title("Distribution of Operation Difficulty")
    ax1.legend()

    # Top 20 hardest operations
    ax2 = axes[0, 1]
    top_k = 20
    hardest_indices = np.argsort(difficulties)[-top_k:][::-1]
    hardest_ops = [get_operation_string(idx) for idx in hardest_indices]
    hardest_vals = difficulties[hardest_indices]

    y_pos = np.arange(top_k)
    ax2.barh(y_pos, hardest_vals, color="crimson", alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(hardest_ops, fontsize=8, fontfamily="monospace")
    ax2.set_xlabel("Difficulty")
    ax2.set_title(f"Top {top_k} Hardest Operations")
    ax2.invert_yaxis()

    # Difficulty by operation property
    ax3 = axes[1, 0]

    # Compute properties for analysis
    def is_symmetric(idx):
        """Check if op(a,b) = op(b,a)"""
        digits = []
        temp = idx
        for _ in range(9):
            digits.append(temp % 3)
            temp //= 3
        # Symmetric if output same when inputs swapped
        # Input (0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)
        # Indices:  0,    1,    2,    3,    4,    5,    6,    7,    8
        return digits[1] == digits[3] and digits[2] == digits[6] and digits[5] == digits[7]

    def is_zero_preserving(idx):
        """Check if op(0,0) = 0"""
        return idx % 3 == 0

    symmetric_mask = np.array([is_symmetric(i) for i in range(len(difficulties))])
    zero_pres_mask = np.array([is_zero_preserving(i) for i in range(len(difficulties))])

    categories = ["Symmetric", "Non-Symmetric", "Zero-Pres", "Non-Zero-Pres"]
    means = [
        difficulties[symmetric_mask].mean(),
        difficulties[~symmetric_mask].mean(),
        difficulties[zero_pres_mask].mean(),
        difficulties[~zero_pres_mask].mean(),
    ]
    stds = [
        difficulties[symmetric_mask].std(),
        difficulties[~symmetric_mask].std(),
        difficulties[zero_pres_mask].std(),
        difficulties[~zero_pres_mask].std(),
    ]

    x_pos = np.arange(len(categories))
    ax3.bar(
        x_pos,
        means,
        yerr=stds,
        color=["blue", "orange", "green", "red"],
        alpha=0.7,
        capsize=5,
    )
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(categories)
    ax3.set_ylabel("Mean Difficulty")
    ax3.set_title("Difficulty by Operation Property")

    # Difficulty in latent space
    ax4 = axes[1, 1]
    z_np = latent_codes.numpy()
    z_centered = z_np - z_np.mean(axis=0)
    U, S, Vt = np.linalg.svd(z_centered, full_matrices=False)
    z_2d = z_centered @ Vt[:2].T

    scatter = ax4.scatter(z_2d[:, 0], z_2d[:, 1], c=difficulties, cmap="hot", s=2, alpha=0.6)
    ax4.set_xlabel("PC1")
    ax4.set_ylabel("PC2")
    ax4.set_title("Difficulty in Latent Space\n(Yellow = Hard)")
    fig.colorbar(scatter, ax=ax4, label="Difficulty")

    plt.tight_layout()
    plt.savefig(output_dir / "operation_difficulty.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Difficulty range: [{difficulties.min():.6f}, {difficulties.max():.6f}]")
    print(f"  Mean difficulty: {difficulties.mean():.6f}")
    print(f"  Std difficulty: {difficulties.std():.6f}")
    print(f"  Hardest operation: {get_operation_string(hardest_indices[0])} (loss={hardest_vals[0]:.6f})")
    print(f"  Symmetric ops mean: {means[0]:.6f}, Non-symmetric: {means[1]:.6f}")
    print("  Saved: operation_difficulty.png")

    # Save summary statistics
    summary = {
        "jacobian": {
            "min": float(jacobian_norms.min()),
            "max": float(jacobian_norms.max()),
            "mean": float(jacobian_norms.mean()),
        },
        "holes": {
            "fraction_percent": float(hole_fraction),
            "density_range": [float(density.min()), float(density.max())],
        },
        "difficulty": {
            "min": float(difficulties.min()),
            "max": float(difficulties.max()),
            "mean": float(difficulties.mean()),
            "std": float(difficulties.std()),
            "top_5_hardest": [
                {
                    "op": get_operation_string(idx),
                    "loss": float(difficulties[idx]),
                }
                for idx in hardest_indices[:5]
            ],
            "symmetric_mean": float(means[0]),
            "non_symmetric_mean": float(means[1]),
            "zero_preserving_mean": float(means[2]),
            "non_zero_preserving_mean": float(means[3]),
        },
    }

    with open(output_dir / "advanced_manifold_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n  Summary saved: advanced_manifold_summary.json")

    return summary


def main():
    """Main execution."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Find latest checkpoint
    checkpoint_dirs = [
        CHECKPOINTS_DIR / "v5_6",
        CHECKPOINTS_DIR / "v5_5",
        CHECKPOINTS_DIR,
    ]

    checkpoints = []
    for checkpoint_dir in checkpoint_dirs:
        if checkpoint_dir.exists():
            checkpoints.extend(list(checkpoint_dir.glob("*.pt")))

    if not checkpoints:
        print("No checkpoints found!")
        return

    # Prefer 'best.pt' from v5_6
    best_v56 = CHECKPOINTS_DIR / "v5_6" / "best.pt"
    checkpoint_path = best_v56 if best_v56.exists() else max(checkpoints, key=lambda x: x.stat().st_mtime)

    print(f"Loading checkpoint: {checkpoint_path.name}")

    # Load model
    model, config = load_checkpoint(str(checkpoint_path), device=device)
    print(f"Model loaded: latent_dim={config.get('latent_dim', 16)}")

    # Generate and encode operations
    print("\nGenerating all 19,683 ternary operations...")
    operations = generate_all_operations(device=device)

    print("Encoding operations to latent space...")
    latent_codes, mu, logvar = encode_all_operations(model, operations)
    print(f"Encoded shape: {latent_codes.shape}")

    # Run all analyses
    print("\n" + "=" * 60)
    print("ADVANCED MANIFOLD ANALYSIS")
    print("=" * 60)

    plot_all_analyses(model, operations, latent_codes, device=device)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nOutput directory: outputs/manifold_viz/")
    print("Generated files:")
    print("  - jacobian_surface.png")
    print("  - voronoi_cells.png")
    print("  - latent_holes.png")
    print("  - operation_difficulty.png")
    print("  - advanced_manifold_summary.json")


if __name__ == "__main__":
    main()
