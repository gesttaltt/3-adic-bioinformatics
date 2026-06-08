# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Ternary Manifold Surface Visualization (v5.6 and v5.10).

Generates 3D surface plots of:
1. Loss Landscape: z = reconstruction_loss(PC1, PC2) - optimization surface
2. Real Manifold: 19,683 encoded points projected to 3D with fitted surface

Usage:
    python src/scripts/visualization/visualize_ternary_manifold.py --checkpoint latest.pt
    python src/scripts/visualization/visualize_ternary_manifold.py --checkpoint latest.pt --vae B
    python src/scripts/visualization/visualize_ternary_manifold.py --checkpoint latest.pt --model-version v5.10
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import CHECKPOINTS_DIR, VIZ_DIR
from src.data.generation import generate_all_ternary_operations

# V5.11 is the canonical model - imports V5.6/V5.10 aliases for backwards compatibility
from src.models import TernaryVAE as DualNeuralVAEV5
from src.models import TernaryVAE as DualNeuralVAEV5_10


def load_checkpoint(checkpoint_path: Path, device: str = "cpu", model_version: str = "v5.6"):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if model_version == "v5.10":
        # v5.10 model with hyperbolic-aware StateNet (v5.10.1 curriculum support)
        model_config = checkpoint.get("config", {}).get("model", {})
        model = DualNeuralVAEV5_10(
            input_dim=9,
            latent_dim=16,
            use_statenet=model_config.get("use_statenet", True),
            statenet_version=model_config.get("statenet_version", 4),
            statenet_curriculum_scale=model_config.get("statenet_curriculum_scale", 0.1),
        )
        print("Loading v5.10.1 model (Radial-First Curriculum Learning)")
    else:
        # v5.6 model
        use_statenet = checkpoint.get("statenet_enabled", False)
        model = DualNeuralVAEV5(input_dim=9, latent_dim=16, use_statenet=use_statenet)
        print("Loading v5.6 model (Legacy)")

    # Handle different checkpoint formats
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path.name}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")

    if model_version == "v5.10":
        print(f"  Best hyp corr: {checkpoint.get('best_corr_hyp', 'N/A')}")
        print(f"  Best coverage: {checkpoint.get('best_coverage', 'N/A')}")
    else:
        use_statenet = checkpoint.get("statenet_enabled", False)
        print(f"  StateNet: {'enabled' if use_statenet else 'disabled'}")

    # Print coverage history if available
    cov_A = checkpoint.get("coverage_A_history", [])
    cov_B = checkpoint.get("coverage_B_history", [])
    if cov_A:
        print(f"  Coverage A: {cov_A[-1]} ({cov_A[-1] / 19683 * 100:.2f}%)")
    if cov_B:
        print(f"  Coverage B: {cov_B[-1]} ({cov_B[-1] / 19683 * 100:.2f}%)")

    return model


def encode_all_operations(model, device: str = "cpu", use_vae: str = "A"):
    """Encode all 19,683 ternary operations and compute reconstruction losses."""
    operations = generate_all_ternary_operations()
    x = torch.tensor(operations, dtype=torch.float32, device=device)

    with torch.no_grad():
        if use_vae == "A":
            mu, logvar = model.encoder_A(x)
            z = mu  # Use mean for deterministic encoding
            logits = model.decoder_A(z)
        else:
            mu, logvar = model.encoder_B(x)
            z = mu
            logits = model.decoder_B(z)

        # Compute per-sample reconstruction loss (cross-entropy)
        # Shape: [19683, 9, 3] -> [19683]
        targets = (x + 1).long()  # Convert {-1,0,1} to {0,1,2}
        losses = F.cross_entropy(logits.view(-1, 3), targets.view(-1), reduction="none").view(-1, 9).mean(dim=1)

        # Compute reconstruction accuracy per sample
        preds = logits.argmax(dim=-1) - 1  # Back to {-1,0,1}
        accuracy = (preds == x).float().mean(dim=1)

        # Compute KL divergence per sample: KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_total = kl_per_dim.sum(dim=1)  # Total KL per operation

        # Compute latent variance (uncertainty) per sample
        variance = logvar.exp().mean(dim=1)  # Mean variance across dimensions

    return {
        "z": z.cpu().numpy(),
        "mu": mu.cpu().numpy(),
        "logvar": logvar.cpu().numpy(),
        "losses": losses.cpu().numpy(),
        "accuracy": accuracy.cpu().numpy(),
        "kl_divergence": kl_total.cpu().numpy(),
        "kl_per_dim": kl_per_dim.cpu().numpy(),
        "variance": variance.cpu().numpy(),
        "operations": operations,
    }


def compute_operation_properties(operations: np.ndarray) -> dict:
    """Compute semantic properties of ternary operations for clustering."""
    n_ops = len(operations)

    # 1. Symmetry: Check if op(a,b) = op(b,a) for all inputs
    # Input pairs: (i,j) where i,j in {-1,0,1}, flattened as 3*i + j
    symmetry = np.zeros(n_ops)
    for idx, op in enumerate(operations):
        # Compare op[3*i+j] with op[3*j+i] for all i,j
        is_symmetric = True
        for i in range(3):
            for j in range(3):
                if op[3 * i + j] != op[3 * j + i]:
                    is_symmetric = False
                    break
            if not is_symmetric:
                break
        symmetry[idx] = 1.0 if is_symmetric else 0.0

    # 2. Linearity score: How close to a linear function
    # A linear ternary op would satisfy: op(a,b) = c1*a + c2*b (mod 3 shifted)
    linearity = np.zeros(n_ops)
    for idx, op in enumerate(operations):
        # Check deviation from best linear fit
        best_fit = float("inf")
        for c1 in [-1, 0, 1]:
            for c2 in [-1, 0, 1]:
                deviation = 0
                for i, a in enumerate([-1, 0, 1]):
                    for j, b in enumerate([-1, 0, 1]):
                        expected = np.clip(c1 * a + c2 * b, -1, 1)
                        actual = op[3 * i + j]
                        deviation += abs(expected - actual)
                best_fit = min(best_fit, deviation)
        linearity[idx] = 1.0 - best_fit / 18.0  # Normalize to [0,1]

    # 3. Zero-preserving: op(0,0) = 0
    zero_preserving = (operations[:, 4] == 0).astype(float)  # Index 4 = (0,0)

    # 4. Idempotency: op(a,a) = a for all a
    idempotent = np.zeros(n_ops)
    for idx, op in enumerate(operations):
        is_idempotent = (op[0] == -1) and (op[4] == 0) and (op[8] == 1)
        idempotent[idx] = 1.0 if is_idempotent else 0.0

    # 5. Entropy of output distribution
    output_entropy = np.zeros(n_ops)
    for idx, op in enumerate(operations):
        counts = np.array([np.sum(op == v) for v in [-1, 0, 1]])
        probs = counts / 9.0
        probs = probs[probs > 0]
        output_entropy[idx] = -np.sum(probs * np.log2(probs + 1e-10))

    # 6. Balance: How evenly distributed are outputs
    balance = output_entropy / np.log2(3)  # Normalize by max entropy

    return {
        "symmetry": symmetry,
        "linearity": linearity,
        "zero_preserving": zero_preserving,
        "idempotent": idempotent,
        "output_entropy": output_entropy,
        "balance": balance,
    }


def plot_loss_landscape(data: dict, output_path: Path, vae_name: str = "A"):
    """Plot loss landscape as 3D surface using PCA projection."""
    z = data["z"]
    losses = data["losses"]

    # PCA to 2D
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z)

    print(f"PCA explained variance: {pca.explained_variance_ratio_}")
    print(f"  PC1: {pca.explained_variance_ratio_[0] * 100:.1f}%")
    print(f"  PC2: {pca.explained_variance_ratio_[1] * 100:.1f}%")

    # Create dense grid
    grid_res = 100
    x_min, x_max = z_2d[:, 0].min() - 0.5, z_2d[:, 0].max() + 0.5
    y_min, y_max = z_2d[:, 1].min() - 0.5, z_2d[:, 1].max() + 0.5

    xi = np.linspace(x_min, x_max, grid_res)
    yi = np.linspace(y_min, y_max, grid_res)
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolate losses onto grid
    Zi = griddata(z_2d, losses, (Xi, Yi), method="cubic", fill_value=np.nan)

    # Smooth for visualization
    Zi_smooth = gaussian_filter(np.nan_to_num(Zi, nan=np.nanmean(Zi)), sigma=1.5)

    # Create figure
    fig = plt.figure(figsize=(14, 6))

    # Surface plot
    ax1 = fig.add_subplot(121, projection="3d")
    surf = ax1.plot_surface(
        Xi,
        Yi,
        Zi_smooth,
        cmap="viridis",
        edgecolor="none",
        alpha=0.9,
        antialiased=True,
    )
    ax1.set_xlabel("PC1", fontsize=10)
    ax1.set_ylabel("PC2", fontsize=10)
    ax1.set_zlabel("Reconstruction Loss", fontsize=10)
    ax1.set_title(f"VAE-{vae_name} Loss Landscape\n(Optimization Surface)", fontsize=12)
    ax1.view_init(elev=25, azim=45)
    fig.colorbar(surf, ax=ax1, shrink=0.5, label="Loss")

    # Contour plot (top-down view)
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(Xi, Yi, Zi_smooth, levels=50, cmap="viridis")
    ax2.scatter(
        z_2d[:, 0],
        z_2d[:, 1],
        c=losses,
        cmap="viridis",
        s=1,
        alpha=0.3,
        edgecolors="none",
    )
    ax2.set_xlabel("PC1", fontsize=10)
    ax2.set_ylabel("PC2", fontsize=10)
    ax2.set_title(
        f"VAE-{vae_name} Loss Landscape (Top View)\nDots = Encoded Operations",
        fontsize=12,
    )
    fig.colorbar(contour, ax=ax2, label="Loss")

    plt.tight_layout()
    plt.savefig(
        output_path / f"loss_landscape_vae_{vae_name.lower()}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved: {output_path / f'loss_landscape_vae_{vae_name.lower()}.png'}")

    return pca, z_2d


def plot_real_manifold(data: dict, pca: PCA, output_path: Path, vae_name: str = "A"):
    """Plot the real ternary manifold - 19,683 points in 3D latent space."""
    z = data["z"]
    accuracy = data["accuracy"]
    losses = data["losses"]

    # PCA to 3D for the actual manifold
    pca_3d = PCA(n_components=3)
    z_3d = pca_3d.fit_transform(z)

    print("\n3D PCA explained variance:")
    print(f"  PC1: {pca_3d.explained_variance_ratio_[0] * 100:.1f}%")
    print(f"  PC2: {pca_3d.explained_variance_ratio_[1] * 100:.1f}%")
    print(f"  PC3: {pca_3d.explained_variance_ratio_[2] * 100:.1f}%")
    print(f"  Total: {sum(pca_3d.explained_variance_ratio_) * 100:.1f}%")

    # Create figure with multiple views
    fig = plt.figure(figsize=(16, 12))

    # Main 3D scatter - colored by reconstruction accuracy
    ax1 = fig.add_subplot(221, projection="3d")
    scatter1 = ax1.scatter(
        z_3d[:, 0],
        z_3d[:, 1],
        z_3d[:, 2],
        c=accuracy,
        cmap="RdYlGn",
        s=3,
        alpha=0.7,
    )
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_zlabel("PC3")
    ax1.set_title(f"VAE-{vae_name} Ternary Manifold\nColor = Reconstruction Accuracy")
    ax1.view_init(elev=20, azim=45)
    fig.colorbar(scatter1, ax=ax1, shrink=0.5, label="Accuracy")

    # 3D scatter - colored by loss (inverse view for peaks/valleys)
    ax2 = fig.add_subplot(222, projection="3d")
    scatter2 = ax2.scatter(
        z_3d[:, 0],
        z_3d[:, 1],
        z_3d[:, 2],
        c=losses,
        cmap="viridis_r",
        s=3,
        alpha=0.7,
    )
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_zlabel("PC3")
    ax2.set_title(f"VAE-{vae_name} Ternary Manifold\nColor = Reconstruction Loss")
    ax2.view_init(elev=20, azim=135)
    fig.colorbar(scatter2, ax=ax2, shrink=0.5, label="Loss")

    # Fitted surface through point cloud (using interpolation)
    ax3 = fig.add_subplot(223, projection="3d")

    # Create surface by interpolating z_3d[:, 2] over (PC1, PC2) grid
    grid_res = 50
    xi = np.linspace(z_3d[:, 0].min(), z_3d[:, 0].max(), grid_res)
    yi = np.linspace(z_3d[:, 1].min(), z_3d[:, 1].max(), grid_res)
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolate PC3 values
    Zi = griddata(z_3d[:, :2], z_3d[:, 2], (Xi, Yi), method="cubic", fill_value=np.nan)
    Zi_smooth = gaussian_filter(np.nan_to_num(Zi, nan=np.nanmean(Zi)), sigma=1)

    # Surface with points overlay
    ax3.plot_surface(
        Xi,
        Yi,
        Zi_smooth,
        cmap="coolwarm",
        alpha=0.6,
        antialiased=True,
        edgecolor="none",
    )
    ax3.scatter(z_3d[:, 0], z_3d[:, 1], z_3d[:, 2], c="black", s=1, alpha=0.3)
    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")
    ax3.set_zlabel("PC3")
    ax3.set_title(f"VAE-{vae_name} Manifold Surface\n(Fitted through Point Cloud)")
    ax3.view_init(elev=25, azim=60)

    # Density heatmap (2D projection)
    ax4 = fig.add_subplot(224)
    heatmap, xedges, yedges = np.histogram2d(z_3d[:, 0], z_3d[:, 1], bins=50)
    heatmap = gaussian_filter(heatmap.T, sigma=1)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax4.imshow(heatmap, extent=extent, origin="lower", cmap="hot", aspect="auto")
    ax4.set_xlabel("PC1")
    ax4.set_ylabel("PC2")
    ax4.set_title(f"VAE-{vae_name} Manifold Density\n(PC1 vs PC2 Projection)")
    fig.colorbar(im, ax=ax4, label="Density")

    plt.tight_layout()
    plt.savefig(
        output_path / f"real_manifold_vae_{vae_name.lower()}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved: {output_path / f'real_manifold_vae_{vae_name.lower()}.png'}")

    return z_3d


def plot_combined_high_res(data_A: dict, data_B: dict, output_path: Path):
    """Create high-resolution combined visualization of both VAEs."""
    # PCA on combined latent space
    z_combined = np.vstack([data_A["z"], data_B["z"]])
    pca = PCA(n_components=3)
    z_combined_3d = pca.fit_transform(z_combined)

    z_A_3d = z_combined_3d[:19683]
    z_B_3d = z_combined_3d[19683:]

    fig = plt.figure(figsize=(16, 8))

    # VAE-A manifold
    ax1 = fig.add_subplot(121, projection="3d")
    scatter1 = ax1.scatter(
        z_A_3d[:, 0],
        z_A_3d[:, 1],
        z_A_3d[:, 2],
        c=data_A["accuracy"],
        cmap="RdYlGn",
        s=2,
        alpha=0.6,
    )
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_zlabel("PC3")
    ax1.set_title("VAE-A: Chaotic Regime\n(Compact Latent Space)")
    ax1.view_init(elev=20, azim=45)
    fig.colorbar(scatter1, ax=ax1, shrink=0.5, label="Accuracy")

    # VAE-B manifold
    ax2 = fig.add_subplot(122, projection="3d")
    scatter2 = ax2.scatter(
        z_B_3d[:, 0],
        z_B_3d[:, 1],
        z_B_3d[:, 2],
        c=data_B["accuracy"],
        cmap="RdYlGn",
        s=2,
        alpha=0.6,
    )
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_zlabel("PC3")
    ax2.set_title("VAE-B: Frozen Regime\n(Expanded Latent Space)")
    ax2.view_init(elev=20, azim=45)
    fig.colorbar(scatter2, ax=ax2, shrink=0.5, label="Accuracy")

    plt.suptitle(
        "Dual-VAE Ternary Manifold Comparison\n(Same PCA Basis)",
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(
        output_path / "manifold_comparison_dual_vae.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved: {output_path / 'manifold_comparison_dual_vae.png'}")


def plot_kl_divergence_surface(data: dict, output_path: Path, vae_name: str = "A"):
    """Plot KL divergence surface - shows information content per operation."""
    z = data["z"]
    kl = data["kl_divergence"]

    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z)

    # Create grid
    grid_res = 80
    x_min, x_max = z_2d[:, 0].min() - 0.5, z_2d[:, 0].max() + 0.5
    y_min, y_max = z_2d[:, 1].min() - 0.5, z_2d[:, 1].max() + 0.5

    xi = np.linspace(x_min, x_max, grid_res)
    yi = np.linspace(y_min, y_max, grid_res)
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolate KL onto grid
    Zi = griddata(z_2d, kl, (Xi, Yi), method="cubic", fill_value=np.nan)
    Zi_smooth = gaussian_filter(np.nan_to_num(Zi, nan=np.nanmean(Zi)), sigma=1.2)

    fig = plt.figure(figsize=(16, 6))

    # 3D surface
    ax1 = fig.add_subplot(131, projection="3d")
    surf = ax1.plot_surface(
        Xi,
        Yi,
        Zi_smooth,
        cmap="magma",
        edgecolor="none",
        alpha=0.9,
        antialiased=True,
    )
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_zlabel("KL Divergence")
    ax1.set_title(f"VAE-{vae_name} Information Content\nz = KL(operation)")
    ax1.view_init(elev=25, azim=45)
    fig.colorbar(surf, ax=ax1, shrink=0.5, label="KL (nats)")

    # Contour with scatter
    ax2 = fig.add_subplot(132)
    contour = ax2.contourf(Xi, Yi, Zi_smooth, levels=40, cmap="magma")
    ax2.scatter(
        z_2d[:, 0],
        z_2d[:, 1],
        c=kl,
        cmap="magma",
        s=2,
        alpha=0.5,
        edgecolors="none",
    )
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_title(f"VAE-{vae_name} KL Divergence Map\nHigh KL = Rare/Complex Operations")
    fig.colorbar(contour, ax=ax2, label="KL (nats)")

    # Histogram of KL values
    ax3 = fig.add_subplot(133)
    ax3.hist(kl, bins=50, color="purple", alpha=0.7, edgecolor="black")
    ax3.axvline(kl.mean(), color="red", linestyle="--", label=f"Mean: {kl.mean():.2f}")
    ax3.axvline(
        np.median(kl),
        color="orange",
        linestyle="--",
        label=f"Median: {np.median(kl):.2f}",
    )
    ax3.set_xlabel("KL Divergence (nats)")
    ax3.set_ylabel("Count")
    ax3.set_title(f"VAE-{vae_name} KL Distribution\n{len(kl)} operations")
    ax3.legend()

    plt.tight_layout()
    plt.savefig(
        output_path / f"kl_divergence_vae_{vae_name.lower()}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved: {output_path / f'kl_divergence_vae_{vae_name.lower()}.png'}")
    print(f"  KL stats: mean={kl.mean():.3f}, std={kl.std():.3f}, max={kl.max():.3f}")


def plot_variance_surface(data: dict, output_path: Path, vae_name: str = "A"):
    """Plot latent variance surface - shows model uncertainty per operation."""
    z = data["z"]
    variance = data["variance"]
    logvar = data["logvar"]

    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z)

    # Create grid
    grid_res = 80
    x_min, x_max = z_2d[:, 0].min() - 0.5, z_2d[:, 0].max() + 0.5
    y_min, y_max = z_2d[:, 1].min() - 0.5, z_2d[:, 1].max() + 0.5

    xi = np.linspace(x_min, x_max, grid_res)
    yi = np.linspace(y_min, y_max, grid_res)
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolate variance onto grid
    Zi = griddata(z_2d, variance, (Xi, Yi), method="cubic", fill_value=np.nan)
    Zi_smooth = gaussian_filter(np.nan_to_num(Zi, nan=np.nanmean(Zi)), sigma=1.2)

    fig = plt.figure(figsize=(16, 10))

    # 3D surface
    ax1 = fig.add_subplot(221, projection="3d")
    surf = ax1.plot_surface(
        Xi,
        Yi,
        Zi_smooth,
        cmap="plasma",
        edgecolor="none",
        alpha=0.9,
        antialiased=True,
    )
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_zlabel("Variance")
    ax1.set_title(f"VAE-{vae_name} Uncertainty Surface\nz = mean(exp(logvar))")
    ax1.view_init(elev=25, azim=45)
    fig.colorbar(surf, ax=ax1, shrink=0.5, label="Variance")

    # Contour
    ax2 = fig.add_subplot(222)
    contour = ax2.contourf(Xi, Yi, Zi_smooth, levels=40, cmap="plasma")
    ax2.scatter(
        z_2d[:, 0],
        z_2d[:, 1],
        c=variance,
        cmap="plasma",
        s=2,
        alpha=0.5,
        edgecolors="none",
    )
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_title(f"VAE-{vae_name} Uncertainty Map\nHigh = Model Uncertain")
    fig.colorbar(contour, ax=ax2, label="Variance")

    # Per-dimension variance heatmap
    ax3 = fig.add_subplot(223)
    var_per_dim = np.exp(logvar)  # Shape: [19683, 16]
    im = ax3.imshow(var_per_dim[:100].T, aspect="auto", cmap="plasma")
    ax3.set_xlabel("Operation Index (first 100)")
    ax3.set_ylabel("Latent Dimension")
    ax3.set_title(f"VAE-{vae_name} Per-Dimension Variance\n(First 100 Operations)")
    fig.colorbar(im, ax=ax3, label="Variance")

    # Variance distribution per dimension
    ax4 = fig.add_subplot(224)
    dim_variances = var_per_dim.mean(axis=0)
    ax4.bar(range(16), dim_variances, color="purple", alpha=0.7)
    ax4.axhline(
        dim_variances.mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {dim_variances.mean():.3f}",
    )
    ax4.set_xlabel("Latent Dimension")
    ax4.set_ylabel("Mean Variance")
    ax4.set_title(f"VAE-{vae_name} Variance by Dimension\n(Active vs Inactive Dimensions)")
    ax4.legend()

    plt.tight_layout()
    plt.savefig(
        output_path / f"variance_surface_vae_{vae_name.lower()}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved: {output_path / f'variance_surface_vae_{vae_name.lower()}.png'}")
    print(f"  Variance stats: mean={variance.mean():.4f}, std={variance.std():.4f}")


def plot_operation_clustering(data: dict, props: dict, output_path: Path, vae_name: str = "A"):
    """Plot manifold colored by operation properties - reveals semantic structure."""
    z = data["z"]

    pca = PCA(n_components=3)
    z_3d = pca.fit_transform(z)

    fig = plt.figure(figsize=(20, 15))

    properties = [
        ("symmetry", "Symmetry", "op(a,b) = op(b,a)", "RdYlGn"),
        ("linearity", "Linearity", "How linear the operation is", "viridis"),
        ("zero_preserving", "Zero-Preserving", "op(0,0) = 0", "coolwarm"),
        (
            "balance",
            "Output Balance",
            "Entropy of output distribution",
            "plasma",
        ),
        ("idempotent", "Idempotent", "op(a,a) = a", "PiYG"),
        ("output_entropy", "Output Entropy", "H(outputs)", "inferno"),
    ]

    for idx, (prop_key, title, desc, cmap) in enumerate(properties):
        ax = fig.add_subplot(2, 3, idx + 1, projection="3d")
        prop_vals = props[prop_key]

        scatter = ax.scatter(
            z_3d[:, 0],
            z_3d[:, 1],
            z_3d[:, 2],
            c=prop_vals,
            cmap=cmap,
            s=3,
            alpha=0.6,
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title(f"{title}\n{desc}")
        ax.view_init(elev=20, azim=45 + idx * 15)
        fig.colorbar(scatter, ax=ax, shrink=0.5, label=prop_key)

    plt.suptitle(
        f"VAE-{vae_name} Manifold Colored by Operation Properties\n(Semantic Structure of the Ternary Space)",
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(
        output_path / f"operation_clustering_vae_{vae_name.lower()}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved: {output_path / f'operation_clustering_vae_{vae_name.lower()}.png'}")
    print(f"  Symmetric ops: {int(props['symmetry'].sum())} ({props['symmetry'].mean() * 100:.1f}%)")
    print(f"  Zero-preserving: {int(props['zero_preserving'].sum())} ({props['zero_preserving'].mean() * 100:.1f}%)")
    print(f"  Idempotent: {int(props['idempotent'].sum())} ({props['idempotent'].mean() * 100:.1f}%)")


def plot_semantic_surface(data: dict, props: dict, output_path: Path, vae_name: str = "A"):
    """Plot 3D surface where height = semantic property value."""
    z = data["z"]

    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z)

    # Create grid
    grid_res = 60
    x_min, x_max = z_2d[:, 0].min() - 0.5, z_2d[:, 0].max() + 0.5
    y_min, y_max = z_2d[:, 1].min() - 0.5, z_2d[:, 1].max() + 0.5

    xi = np.linspace(x_min, x_max, grid_res)
    yi = np.linspace(y_min, y_max, grid_res)
    Xi, Yi = np.meshgrid(xi, yi)

    fig = plt.figure(figsize=(16, 12))

    surfaces = [
        ("linearity", "Linearity Surface", "coolwarm"),
        ("balance", "Output Balance Surface", "viridis"),
        ("output_entropy", "Output Entropy Surface", "plasma"),
    ]

    for idx, (prop_key, title, cmap) in enumerate(surfaces):
        prop_vals = props[prop_key]

        # Interpolate property onto grid
        Zi = griddata(z_2d, prop_vals, (Xi, Yi), method="cubic", fill_value=np.nan)
        Zi_smooth = gaussian_filter(np.nan_to_num(Zi, nan=np.nanmean(Zi)), sigma=1.5)

        # 3D surface
        ax1 = fig.add_subplot(2, 3, idx + 1, projection="3d")
        ax1.plot_surface(
            Xi,
            Yi,
            Zi_smooth,
            cmap=cmap,
            edgecolor="none",
            alpha=0.9,
            antialiased=True,
        )
        ax1.set_xlabel("PC1")
        ax1.set_ylabel("PC2")
        ax1.set_zlabel(prop_key.capitalize())
        ax1.set_title(f"VAE-{vae_name} {title}")
        ax1.view_init(elev=30, azim=45)

        # Contour
        ax2 = fig.add_subplot(2, 3, idx + 4)
        contour = ax2.contourf(Xi, Yi, Zi_smooth, levels=30, cmap=cmap)
        ax2.scatter(
            z_2d[:, 0],
            z_2d[:, 1],
            c=prop_vals,
            cmap=cmap,
            s=1,
            alpha=0.3,
            edgecolors="none",
        )
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        ax2.set_title(f"{title} (Top View)")
        fig.colorbar(contour, ax=ax2, label=prop_key)

    plt.tight_layout()
    plt.savefig(
        output_path / f"semantic_surfaces_vae_{vae_name.lower()}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved: {output_path / f'semantic_surfaces_vae_{vae_name.lower()}.png'}")


def plot_manifold_with_gradient_flow(data: dict, output_path: Path, vae_name: str = "A"):
    """Plot manifold surface with gradient flow visualization."""
    z = data["z"]
    losses = data["losses"]

    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z)

    # Create grid
    grid_res = 60
    x_min, x_max = z_2d[:, 0].min() - 0.5, z_2d[:, 0].max() + 0.5
    y_min, y_max = z_2d[:, 1].min() - 0.5, z_2d[:, 1].max() + 0.5

    xi = np.linspace(x_min, x_max, grid_res)
    yi = np.linspace(y_min, y_max, grid_res)
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolate and smooth
    Zi = griddata(z_2d, losses, (Xi, Yi), method="cubic", fill_value=np.nan)
    Zi = np.nan_to_num(Zi, nan=np.nanmean(Zi))
    Zi_smooth = gaussian_filter(Zi, sigma=1.5)

    # Compute gradients for flow
    dZdx, dZdy = np.gradient(Zi_smooth)

    fig = plt.figure(figsize=(14, 10))

    # 3D surface with wireframe
    ax1 = fig.add_subplot(221, projection="3d")
    ax1.plot_surface(
        Xi,
        Yi,
        Zi_smooth,
        cmap="viridis",
        alpha=0.8,
        antialiased=True,
        edgecolor="none",
    )
    ax1.plot_wireframe(
        Xi[::5, ::5],
        Yi[::5, ::5],
        Zi_smooth[::5, ::5],
        color="black",
        alpha=0.2,
        linewidth=0.5,
    )
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_zlabel("Loss")
    ax1.set_title(f"VAE-{vae_name} Loss Landscape\nwith Mesh Overlay")
    ax1.view_init(elev=30, azim=45)

    # Contour with gradient flow
    ax2 = fig.add_subplot(222)
    contour = ax2.contourf(Xi, Yi, Zi_smooth, levels=30, cmap="viridis")
    # Quiver plot for gradient flow (every 4th point)
    skip = 4
    ax2.quiver(
        Xi[::skip, ::skip],
        Yi[::skip, ::skip],
        -dZdx[::skip, ::skip],
        -dZdy[::skip, ::skip],
        color="white",
        alpha=0.6,
        scale=50,
    )
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_title(f"VAE-{vae_name} Gradient Flow\n(Arrows point toward lower loss)")
    fig.colorbar(contour, ax=ax2, label="Loss")

    # Saddle point detection (where gradient magnitude is low but curvature varies)
    grad_mag = np.sqrt(dZdx**2 + dZdy**2)
    ax3 = fig.add_subplot(223)
    im3 = ax3.imshow(
        grad_mag,
        extent=[x_min, x_max, y_min, y_max],
        origin="lower",
        cmap="plasma",
        aspect="auto",
    )
    ax3.contour(Xi, Yi, Zi_smooth, levels=15, colors="white", alpha=0.5, linewidths=0.5)
    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")
    ax3.set_title(f"VAE-{vae_name} Gradient Magnitude\n(Low = Valleys/Saddles)")
    fig.colorbar(im3, ax=ax3, label="|∇Loss|")

    # Curvature approximation (Laplacian)
    from scipy.ndimage import laplace

    curvature = laplace(Zi_smooth)
    ax4 = fig.add_subplot(224)
    im4 = ax4.imshow(
        curvature,
        extent=[x_min, x_max, y_min, y_max],
        origin="lower",
        cmap="RdBu",
        aspect="auto",
        vmin=-np.percentile(np.abs(curvature), 95),
        vmax=np.percentile(np.abs(curvature), 95),
    )
    ax4.scatter(z_2d[:, 0], z_2d[:, 1], c="black", s=0.5, alpha=0.2)
    ax4.set_xlabel("PC1")
    ax4.set_ylabel("PC2")
    ax4.set_title(f"VAE-{vae_name} Curvature (Laplacian)\nRed=Peaks, Blue=Valleys")
    fig.colorbar(im4, ax=ax4, label="∇²Loss")

    plt.tight_layout()
    plt.savefig(
        output_path / f"gradient_flow_vae_{vae_name.lower()}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved: {output_path / f'gradient_flow_vae_{vae_name.lower()}.png'}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Ternary Manifold (v5.6 or v5.10)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="latest.pt",
        help="Checkpoint filename or full path",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="v5.6",
        choices=["v5.6", "v5.10"],
        help="Model version (default: v5.6)",
    )
    parser.add_argument(
        "--vae",
        type=str,
        default="both",
        choices=["A", "B", "both"],
        help="Which VAE to visualize",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(VIZ_DIR / "manifold_viz"),
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()

    # Setup paths - support both relative checkpoint names and full paths
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_dir = CHECKPOINTS_DIR / "v5_10" if args.model_version == "v5.10" else CHECKPOINTS_DIR / "v5_5"
        checkpoint_path = checkpoint_dir / args.checkpoint

    output_path = PROJECT_ROOT / args.output
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Model version: {args.model_version}")
    print(f"Device: {args.device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_path}")
    print()

    # Load model
    model = load_checkpoint(checkpoint_path, args.device, args.model_version)

    # Compute operation properties once (shared across VAEs)
    print("\n" + "=" * 60)
    print("Computing Operation Properties")
    print("=" * 60)
    from src.data.generation import generate_all_ternary_operations

    operations = generate_all_ternary_operations()
    props = compute_operation_properties(operations)
    print(f"Computed properties for {len(operations)} operations")

    # Process VAE(s)
    if args.vae in ["A", "both"]:
        print("\n" + "=" * 60)
        print("Processing VAE-A (Chaotic Regime)")
        print("=" * 60)
        data_A = encode_all_operations(model, args.device, use_vae="A")
        print(f"Encoded {len(data_A['z'])} operations")
        print(f"Mean accuracy: {data_A['accuracy'].mean() * 100:.2f}%")
        print(f"Mean loss: {data_A['losses'].mean():.4f}")
        print(f"Mean KL: {data_A['kl_divergence'].mean():.4f}")

        pca_A, _ = plot_loss_landscape(data_A, output_path, "A")
        plot_real_manifold(data_A, pca_A, output_path, "A")
        plot_manifold_with_gradient_flow(data_A, output_path, "A")

        # New semantic visualizations
        print("\n--- Semantic Visualizations (VAE-A) ---")
        plot_kl_divergence_surface(data_A, output_path, "A")
        plot_variance_surface(data_A, output_path, "A")
        plot_operation_clustering(data_A, props, output_path, "A")
        plot_semantic_surface(data_A, props, output_path, "A")

    if args.vae in ["B", "both"]:
        print("\n" + "=" * 60)
        print("Processing VAE-B (Frozen Regime)")
        print("=" * 60)
        data_B = encode_all_operations(model, args.device, use_vae="B")
        print(f"Encoded {len(data_B['z'])} operations")
        print(f"Mean accuracy: {data_B['accuracy'].mean() * 100:.2f}%")
        print(f"Mean loss: {data_B['losses'].mean():.4f}")
        print(f"Mean KL: {data_B['kl_divergence'].mean():.4f}")

        pca_B, _ = plot_loss_landscape(data_B, output_path, "B")
        plot_real_manifold(data_B, pca_B, output_path, "B")
        plot_manifold_with_gradient_flow(data_B, output_path, "B")

        # New semantic visualizations
        print("\n--- Semantic Visualizations (VAE-B) ---")
        plot_kl_divergence_surface(data_B, output_path, "B")
        plot_variance_surface(data_B, output_path, "B")
        plot_operation_clustering(data_B, props, output_path, "B")
        plot_semantic_surface(data_B, props, output_path, "B")

    if args.vae == "both":
        print("\n" + "=" * 60)
        print("Creating Combined Visualization")
        print("=" * 60)
        plot_combined_high_res(data_A, data_B, output_path)

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"\nOutput files in: {output_path}")
    print("\nGenerated files:")
    for f in sorted(output_path.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
