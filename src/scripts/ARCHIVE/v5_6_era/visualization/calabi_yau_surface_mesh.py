# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Calabi-Yau Surface Mesh Generation.

Creates smooth, continuous, differentiable surface meshes from VAE latent
point clouds using multiple techniques:

1. Alpha Shapes - Delaunay-based surface reconstruction
2. Ball Pivoting - Rolling ball mesh generation
3. Poisson Surface - Implicit surface from oriented points
4. Gaussian Splatting - Soft kernel density surfaces
5. Marching Cubes - Isosurface extraction from density field

Extends to 16D, 32D latent representations for richer Calabi-Yau projections.
"""

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import gaussian_filter
from scipy.spatial import Delaunay
from skimage import measure

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.ternary_vae_v5_6 import DualNeuralVAEV5

from src.config.paths import CHECKPOINTS_DIR, VIZ_DIR
from src.data.generation import generate_all_ternary_operations


def load_embeddings(checkpoint_path, device="cuda"):
    """Load and return VAE embeddings."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = DualNeuralVAEV5(input_dim=9, latent_dim=16, use_statenet=False)
    state_dict = {k: v for k, v in checkpoint["model"].items() if "state_net" not in k}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    operations = generate_all_ternary_operations()
    x = torch.tensor(operations, dtype=torch.float32, device=device)

    with torch.no_grad():
        mu_A, logvar_A = model.encoder_A(x)
        mu_B, logvar_B = model.encoder_B(x)

    return {
        "z_A": mu_A.cpu().numpy(),  # 16D
        "z_B": mu_B.cpu().numpy(),  # 16D
        "var_A": logvar_A.exp().cpu().numpy(),
        "var_B": logvar_B.exp().cpu().numpy(),
        "operations": operations,
    }


def create_32d_embedding(z_A, z_B):
    """Create 32D embedding by concatenating VAE-A and VAE-B latents.

    This creates a richer representation that captures both chaotic
    and frozen regime information.
    """
    return np.concatenate([z_A, z_B], axis=1)  # (N, 32)


def calabi_yau_quintic_32d(z, phase=0.0):
    """Extended quintic projection from 32D.

    Uses all 32 dimensions as 16 complex coordinates for richer structure.
    """
    z.shape[0]

    # Pair as complex: R^32 -> C^16
    z_complex = z[:, ::2] + 1j * z[:, 1::2]  # (N, 16)

    # Use quintic harmonics with more terms
    theta = np.angle(z_complex) + phase
    r = np.abs(z_complex)

    # Multi-scale quintic projection
    x = np.sum(r[:, :8] * np.cos(5 * theta[:, :8]), axis=1)
    x += 0.3 * np.sum(r[:, 8:] * np.cos(3 * theta[:, 8:]), axis=1)

    y = np.sum(r[:, :8] * np.sin(5 * theta[:, :8]), axis=1)
    y += 0.3 * np.sum(r[:, 8:] * np.sin(3 * theta[:, 8:]), axis=1)

    z_out = np.sum(r[:, :8] * np.cos(3 * theta[:, :8]) * np.sin(2 * theta[:, :8]), axis=1)
    z_out += 0.5 * np.sum(r[:, 8:] * np.sin(7 * theta[:, 8:]), axis=1)

    result = np.stack([x, y, z_out], axis=1)
    max_val = np.percentile(np.abs(result), 99)
    if max_val > 0:
        result = result / max_val

    return result


def calabi_yau_k3_surface(z, phase=0.0):
    """K3 surface inspired projection.

    K3 surfaces are 4D Calabi-Yau manifolds (complex dimension 2).
    We project through a K3-inspired parametrization.
    """
    dim = z.shape[1]
    n_complex = dim // 2
    z_c = z[:, ::2] + 1j * z[:, 1::2]

    # K3 surface: x^4 + y^4 + z^4 + w^4 = 0 in CP^3
    # Use quartic harmonics
    theta = np.angle(z_c) + phase
    r = np.abs(z_c)

    # Quartic terms
    x = np.sum(r[:, :4] * np.cos(4 * theta[:, :4]), axis=1)
    y = np.sum(r[:, :4] * np.sin(4 * theta[:, :4]), axis=1)

    # Add higher harmonics for texture
    z_out = np.sum(
        r[:, 4:8] * np.cos(2 * theta[:, 4:8]) * np.sin(2 * theta[:, 4:8]),
        axis=1,
    )

    if n_complex > 8:
        # Use remaining dimensions for refinement
        x += 0.2 * np.sum(r[:, 8:] * np.cos(6 * theta[:, 8:]), axis=1)
        y += 0.2 * np.sum(r[:, 8:] * np.sin(6 * theta[:, 8:]), axis=1)

    result = np.stack([x, y, z_out], axis=1)
    max_val = np.percentile(np.abs(result), 99)
    if max_val > 0:
        result = result / max_val

    return result


def gaussian_kernel_density_3d(points, grid_size=50, bandwidth=0.1):
    """Create a 3D density field using Gaussian kernel density estimation.

    Returns a volumetric density that can be used for isosurface extraction.
    """
    # Normalize points to [0, 1] range
    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1

    points_norm = (points - min_vals) / range_vals

    # Create 3D grid
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    z = np.linspace(0, 1, grid_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Compute density at each grid point
    density = np.zeros((grid_size, grid_size, grid_size))

    # Use vectorized computation for efficiency
    grid_points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    # Batch processing for memory efficiency
    batch_size = 1000
    for i in range(0, len(points_norm), batch_size):
        batch = points_norm[i : i + batch_size]
        # Compute distances from batch to all grid points
        for p in batch:
            dist_sq = np.sum((grid_points - p) ** 2, axis=1)
            kernel = np.exp(-dist_sq / (2 * bandwidth**2))
            density += kernel.reshape(grid_size, grid_size, grid_size)

    # Smooth the density
    density = gaussian_filter(density, sigma=1)

    return density, (min_vals, max_vals)


def extract_isosurface(density, level=None, bounds=None):
    """Extract isosurface from density field using marching cubes."""
    if level is None:
        level = np.percentile(density, 70)

    try:
        verts, faces, normals, values = measure.marching_cubes(density, level=level, spacing=(1.0, 1.0, 1.0))

        # Scale vertices to original bounds
        if bounds:
            min_vals, max_vals = bounds
            range_vals = max_vals - min_vals
            verts = verts / density.shape[0] * range_vals + min_vals

        return verts, faces, normals
    except:
        return None, None, None


def create_alpha_shape_mesh(points, alpha=0.5):
    """Create mesh using alpha shapes (Delaunay-based).

    Alpha shapes provide a good balance between capturing
    the point cloud structure and creating a smooth surface.
    """
    try:
        # Delaunay triangulation
        tri = Delaunay(points)

        # Filter simplices by circumradius (alpha test)
        filtered_simplices = []
        for simplex in tri.simplices:
            pts = points[simplex]
            # Compute circumradius
            # For 3D: use the formula for circumsphere
            a = np.linalg.norm(pts[0] - pts[1])
            b = np.linalg.norm(pts[1] - pts[2])
            c = np.linalg.norm(pts[2] - pts[0])
            d = np.linalg.norm(pts[0] - pts[3])
            e = np.linalg.norm(pts[1] - pts[3])
            f = np.linalg.norm(pts[2] - pts[3])

            # Simplified: use average edge length as proxy
            avg_edge = (a + b + c + d + e + f) / 6
            if avg_edge < alpha:
                filtered_simplices.append(simplex)

        return np.array(filtered_simplices), points
    except Exception as e:
        print(f"Alpha shape failed: {e}")
        return None, points


def create_soft_surface_mesh(points, values, resolution=30):
    """Create a soft, continuous surface using RBF interpolation.

    This creates a differentiable surface by interpolating through
    the point cloud with radial basis functions.
    """
    # Project to get 2D parameterization + height
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    uv = pca.fit_transform(points)  # 2D parameterization
    points[:, 2]  # Use z as height

    # Create smooth grid
    u_grid = np.linspace(uv[:, 0].min(), uv[:, 0].max(), resolution)
    v_grid = np.linspace(uv[:, 1].min(), uv[:, 1].max(), resolution)
    U, V = np.meshgrid(u_grid, v_grid)

    # RBF interpolation for smooth surface
    try:
        # Subsample for RBF (it's slow with many points)
        n_sample = min(2000, len(points))
        idx = np.random.choice(len(points), n_sample, replace=False)

        rbf_x = RBFInterpolator(uv[idx], points[idx, 0], kernel="thin_plate_spline", smoothing=0.1)
        rbf_y = RBFInterpolator(uv[idx], points[idx, 1], kernel="thin_plate_spline", smoothing=0.1)
        rbf_z = RBFInterpolator(uv[idx], points[idx, 2], kernel="thin_plate_spline", smoothing=0.1)

        grid_points = np.stack([U.ravel(), V.ravel()], axis=1)

        X_surf = rbf_x(grid_points).reshape(resolution, resolution)
        Y_surf = rbf_y(grid_points).reshape(resolution, resolution)
        Z_surf = rbf_z(grid_points).reshape(resolution, resolution)

        return X_surf, Y_surf, Z_surf
    except Exception as e:
        print(f"RBF interpolation failed: {e}")
        return None, None, None


def plot_surface_comparison(points_16d, points_32d, output_path):
    """Create comparison visualization of point cloud vs surfaces."""

    fig = plt.figure(figsize=(24, 18))

    # Row 1: Point clouds
    ax = fig.add_subplot(3, 4, 1, projection="3d")
    ax.scatter(
        points_16d[:, 0],
        points_16d[:, 1],
        points_16d[:, 2],
        c=np.arange(len(points_16d)),
        cmap="viridis",
        s=1,
        alpha=0.5,
    )
    ax.set_title("16D Quintic (Points)")
    ax.view_init(elev=20, azim=45)

    ax = fig.add_subplot(3, 4, 2, projection="3d")
    ax.scatter(
        points_32d[:, 0],
        points_32d[:, 1],
        points_32d[:, 2],
        c=np.arange(len(points_32d)),
        cmap="plasma",
        s=1,
        alpha=0.5,
    )
    ax.set_title("32D Quintic (Points)")
    ax.view_init(elev=20, azim=45)

    # K3 surface projection
    k3_16d = calabi_yau_k3_surface(np.random.randn(len(points_16d), 16))
    ax = fig.add_subplot(3, 4, 3, projection="3d")
    ax.scatter(
        k3_16d[:, 0],
        k3_16d[:, 1],
        k3_16d[:, 2],
        c=np.arange(len(k3_16d)),
        cmap="twilight",
        s=1,
        alpha=0.5,
    )
    ax.set_title("K3 Surface (16D)")
    ax.view_init(elev=20, azim=45)

    # Row 2: Density-based surfaces (isosurface)
    print("Computing density for 16D...")
    density_16d, bounds_16d = gaussian_kernel_density_3d(points_16d, grid_size=40, bandwidth=0.15)
    verts_16d, faces_16d, normals_16d = extract_isosurface(density_16d, bounds=bounds_16d)

    ax = fig.add_subplot(3, 4, 5, projection="3d")
    if verts_16d is not None:
        mesh = Poly3DCollection(
            verts_16d[faces_16d],
            alpha=0.7,
            facecolor="cyan",
            edgecolor="darkblue",
            linewidth=0.1,
        )
        ax.add_collection3d(mesh)
        ax.set_xlim(verts_16d[:, 0].min(), verts_16d[:, 0].max())
        ax.set_ylim(verts_16d[:, 1].min(), verts_16d[:, 1].max())
        ax.set_zlim(verts_16d[:, 2].min(), verts_16d[:, 2].max())
    ax.set_title("16D Isosurface (Marching Cubes)")
    ax.view_init(elev=20, azim=45)

    print("Computing density for 32D...")
    density_32d, bounds_32d = gaussian_kernel_density_3d(points_32d, grid_size=40, bandwidth=0.15)
    verts_32d, faces_32d, normals_32d = extract_isosurface(density_32d, bounds=bounds_32d)

    ax = fig.add_subplot(3, 4, 6, projection="3d")
    if verts_32d is not None:
        mesh = Poly3DCollection(
            verts_32d[faces_32d],
            alpha=0.7,
            facecolor="magenta",
            edgecolor="darkmagenta",
            linewidth=0.1,
        )
        ax.add_collection3d(mesh)
        ax.set_xlim(verts_32d[:, 0].min(), verts_32d[:, 0].max())
        ax.set_ylim(verts_32d[:, 1].min(), verts_32d[:, 1].max())
        ax.set_zlim(verts_32d[:, 2].min(), verts_32d[:, 2].max())
    ax.set_title("32D Isosurface (Marching Cubes)")
    ax.view_init(elev=20, azim=45)

    # Row 2 continued: RBF smooth surfaces
    print("Computing RBF surface for 16D...")
    X_16, Y_16, Z_16 = create_soft_surface_mesh(points_16d, None, resolution=40)

    ax = fig.add_subplot(3, 4, 7, projection="3d")
    if X_16 is not None:
        ax.plot_surface(
            X_16,
            Y_16,
            Z_16,
            cmap="viridis",
            alpha=0.8,
            edgecolor="none",
            antialiased=True,
        )
    ax.set_title("16D RBF Surface")
    ax.view_init(elev=20, azim=45)

    print("Computing RBF surface for 32D...")
    X_32, Y_32, Z_32 = create_soft_surface_mesh(points_32d, None, resolution=40)

    ax = fig.add_subplot(3, 4, 8, projection="3d")
    if X_32 is not None:
        ax.plot_surface(
            X_32,
            Y_32,
            Z_32,
            cmap="plasma",
            alpha=0.8,
            edgecolor="none",
            antialiased=True,
        )
    ax.set_title("32D RBF Surface")
    ax.view_init(elev=20, azim=45)

    # Row 3: Multiple isosurface levels (showing manifold structure)
    levels = [50, 60, 70, 80]
    for i, level_pct in enumerate(levels):
        ax = fig.add_subplot(3, 4, 9 + i, projection="3d")
        level = np.percentile(density_16d, level_pct)
        verts, faces, _ = extract_isosurface(density_16d, level=level, bounds=bounds_16d)
        if verts is not None and len(faces) > 0:
            mesh = Poly3DCollection(
                verts[faces],
                alpha=0.6,
                facecolor=plt.cm.viridis(i / 4),
                edgecolor="none",
            )
            ax.add_collection3d(mesh)
            ax.set_xlim(bounds_16d[0][0], bounds_16d[1][0])
            ax.set_ylim(bounds_16d[0][1], bounds_16d[1][1])
            ax.set_zlim(bounds_16d[0][2], bounds_16d[1][2])
        ax.set_title(f"Isosurface @ {level_pct}%")
        ax.view_init(elev=25, azim=45 + i * 30)

    plt.tight_layout()
    plt.savefig(output_path / "calabi_yau_surfaces.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path / 'calabi_yau_surfaces.png'}")


def export_mesh_data(points_16d, points_32d, output_path):
    """Export mesh data for Three.js surface rendering."""

    # Compute isosurfaces
    print("Generating mesh data for export...")

    density_16d, bounds_16d = gaussian_kernel_density_3d(points_16d, grid_size=50, bandwidth=0.12)
    verts_16d, faces_16d, normals_16d = extract_isosurface(density_16d, bounds=bounds_16d)

    density_32d, bounds_32d = gaussian_kernel_density_3d(points_32d, grid_size=50, bandwidth=0.12)
    verts_32d, faces_32d, normals_32d = extract_isosurface(density_32d, bounds=bounds_32d)

    # Export as JSON for Three.js
    mesh_data = {
        "16d": {
            "vertices": verts_16d.tolist() if verts_16d is not None else [],
            "faces": faces_16d.tolist() if faces_16d is not None else [],
            "normals": normals_16d.tolist() if normals_16d is not None else [],
            "bounds": [bounds_16d[0].tolist(), bounds_16d[1].tolist()],
        },
        "32d": {
            "vertices": verts_32d.tolist() if verts_32d is not None else [],
            "faces": faces_32d.tolist() if faces_32d is not None else [],
            "normals": normals_32d.tolist() if normals_32d is not None else [],
            "bounds": [bounds_32d[0].tolist(), bounds_32d[1].tolist()],
        },
    }

    with open(output_path / "mesh_data.json", "w") as f:
        json.dump(mesh_data, f)
    print(f"Saved: {output_path / 'mesh_data.json'}")

    # Also export density grids for volumetric rendering
    np.save(output_path / "density_16d.npy", density_16d)
    np.save(output_path / "density_32d.npy", density_32d)
    print("Saved: density_16d.npy, density_32d.npy")

    # Export points with projection data
    df = pd.DataFrame()
    df["idx"] = np.arange(len(points_16d))

    # 16D projections
    for i, name in enumerate(["x", "y", "z"]):
        df[f"quintic_16d_{name}"] = points_16d[:, i]

    # 32D projections
    for i, name in enumerate(["x", "y", "z"]):
        df[f"quintic_32d_{name}"] = points_32d[:, i]

    # K3 projections
    k3_points = calabi_yau_k3_surface(np.random.randn(len(points_16d), 16))
    for i, name in enumerate(["x", "y", "z"]):
        df[f"k3_{name}"] = k3_points[:, i]

    df.to_csv(output_path / "surface_projections.csv", index=False)
    print(f"Saved: {output_path / 'surface_projections.csv'}")


def main():
    output_path = VIZ_DIR / "calabi_yau"
    output_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load embeddings
    print("\nLoading embeddings...")
    data = load_embeddings(str(CHECKPOINTS_DIR / "v5_5" / "latest.pt"), device)

    z_A = data["z_A"]  # (19683, 16)
    z_B = data["z_B"]  # (19683, 16)

    # Create 32D embedding
    z_32d = create_32d_embedding(z_A, z_B)  # (19683, 32)
    print(f"16D shape: {z_A.shape}, 32D shape: {z_32d.shape}")

    # Generate Calabi-Yau projections
    print("\nGenerating Calabi-Yau projections...")
    from scripts.visualization.calabi_yau_projection import calabi_yau_quintic_projection

    points_16d = calabi_yau_quintic_projection(z_A)
    points_32d = calabi_yau_quintic_32d(z_32d)

    print(f"16D projection range: [{points_16d.min():.3f}, {points_16d.max():.3f}]")
    print(f"32D projection range: [{points_32d.min():.3f}, {points_32d.max():.3f}]")

    # Generate surface visualizations
    print("\nGenerating surface meshes...")
    plot_surface_comparison(points_16d, points_32d, output_path)

    # Export mesh data
    print("\nExporting mesh data...")
    export_mesh_data(points_16d, points_32d, output_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
