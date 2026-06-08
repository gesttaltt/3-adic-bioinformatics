# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Calabi-Yau Fibration Visualization - Protein-Style Rendering.

Renders the internal fibration structure of Calabi-Yau manifolds using
techniques from protein/molecular visualization:

1. Fiber Backbone Tracing - Connect points along fiber directions
2. Tube/Ribbon Rendering - Smooth splines through fibers
3. Secondary Structure - Identify and render helices/sheets
4. Ambient Occlusion - Depth-aware shading
5. Multi-layer Transparency - Show internal folding

The key insight: Calabi-Yau manifolds have fiber bundle structure.
For the quintic, the Hopf fibration gives S¹ fibers we can trace.
"""

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import CubicSpline
from scipy.spatial import KDTree

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.ternary_vae_v5_6 import DualNeuralVAEV5

from src.config.paths import CHECKPOINTS_DIR, VIZ_DIR
from src.data.generation import generate_all_ternary_operations


def load_embeddings(checkpoint_path, device="cuda"):
    """Load VAE embeddings."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = DualNeuralVAEV5(input_dim=9, latent_dim=16, use_statenet=False)
    state_dict = {k: v for k, v in checkpoint["model"].items() if "state_net" not in k}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    operations = generate_all_ternary_operations()
    x = torch.tensor(operations, dtype=torch.float32, device=device)

    with torch.no_grad():
        mu_A, _ = model.encoder_A(x)
        mu_B, _ = model.encoder_B(x)

    return {
        "z_A": mu_A.cpu().numpy(),
        "z_B": mu_B.cpu().numpy(),
        "operations": operations,
    }


def calabi_yau_projection(z, phase=0.0):
    """Project to 3D using quintic Calabi-Yau."""
    z_complex = z[:, ::2] + 1j * z[:, 1::2]
    theta = np.angle(z_complex[:, :5]) + phase
    r = np.abs(z_complex[:, :5])

    x = np.sum(r * np.cos(5 * theta), axis=1)
    y = np.sum(r * np.sin(5 * theta), axis=1)
    z_out = np.sum(r * np.cos(3 * theta) * np.sin(2 * theta), axis=1)

    result = np.stack([x, y, z_out], axis=1)
    max_val = np.percentile(np.abs(result), 99)
    if max_val > 0:
        result = result / max_val
    return result


def compute_3adic_neighbors(n_ops=19683):
    """Compute 3-adic neighbor graph.

    Two operations are neighbors if they differ by exactly one 3-adic digit.
    This defines the natural "backbone" of the ternary operation space.
    """
    neighbors = {i: [] for i in range(n_ops)}

    for idx in range(n_ops):
        # Compute neighbors by changing each digit
        temp = idx
        multiplier = 1
        for pos in range(9):
            digit = temp % 3
            temp //= 3

            # Try other values for this digit
            for new_digit in range(3):
                if new_digit != digit:
                    neighbor = idx + (new_digit - digit) * multiplier
                    if 0 <= neighbor < n_ops:
                        neighbors[idx].append((neighbor, pos))  # Store position too

            multiplier *= 3

    return neighbors


def trace_fiber_paths(points, neighbors, n_fibers=100, fiber_length=50):
    """Trace fiber paths through the manifold.

    Like tracing the backbone of a protein, we follow connected
    paths through the point cloud using the 3-adic neighbor structure.
    """
    n_points = len(points)
    fibers = []
    used = set()

    # Start fibers from random seed points
    np.random.seed(42)
    seeds = np.random.choice(n_points, size=min(n_fibers * 2, n_points), replace=False)

    for seed in seeds:
        if len(fibers) >= n_fibers:
            break
        if seed in used:
            continue

        # Trace fiber in both directions
        fiber = [seed]
        used.add(seed)

        # Forward direction
        current = seed
        for _ in range(fiber_length // 2):
            # Find unvisited neighbor with smallest position change (stay on same fiber)
            best_neighbor = None
            best_score = float("inf")

            for neighbor, pos in neighbors[current]:
                if neighbor not in used:
                    # Prefer neighbors that continue in same "direction"
                    score = pos  # Lower position = more significant digit change
                    if score < best_score:
                        best_score = score
                        best_neighbor = neighbor

            if best_neighbor is None:
                break

            fiber.append(best_neighbor)
            used.add(best_neighbor)
            current = best_neighbor

        # Backward direction
        current = seed
        for _ in range(fiber_length // 2):
            best_neighbor = None
            best_score = float("inf")

            for neighbor, pos in neighbors[current]:
                if neighbor not in used:
                    score = 8 - pos  # Opposite direction preference
                    if score < best_score:
                        best_score = score
                        best_neighbor = neighbor

            if best_neighbor is None:
                break

            fiber.insert(0, best_neighbor)
            used.add(best_neighbor)
            current = best_neighbor

        if len(fiber) >= 5:  # Only keep substantial fibers
            fibers.append(fiber)

    return fibers


def trace_helical_fibers(points, n_helices=50, points_per_helix=100):
    """Trace helical paths through the manifold.

    The Hopf fibration naturally creates helical structures.
    We identify these by following the phase gradient.
    """
    n_points = len(points)
    tree = KDTree(points)

    helices = []
    used = set()

    np.random.seed(123)
    seeds = np.random.choice(n_points, size=min(n_helices * 3, n_points), replace=False)

    for seed in seeds:
        if len(helices) >= n_helices:
            break
        if seed in used:
            continue

        helix = [seed]
        used.add(seed)
        current_pos = points[seed]

        # Compute local "twist" direction from point distribution
        _, neighbors = tree.query(current_pos, k=20)
        neighbor_points = points[neighbors]

        # PCA to find local tangent
        centered = neighbor_points - neighbor_points.mean(axis=0)
        cov = centered.T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        tangent = eigenvectors[:, -1]  # Principal direction

        # Follow the tangent with slight helical twist
        for step in range(points_per_helix):
            # Add helical rotation
            angle = step * 0.1
            rotation = np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1],
                ]
            )
            twisted_tangent = rotation @ tangent

            # Find nearest point in that direction
            search_pos = current_pos + 0.05 * twisted_tangent
            dist, idx = tree.query(search_pos, k=5)

            # Pick first unvisited
            found = False
            for i, neighbor_idx in enumerate(idx):
                if neighbor_idx not in used and dist[i] < 0.2:
                    helix.append(neighbor_idx)
                    used.add(neighbor_idx)
                    current_pos = points[neighbor_idx]
                    found = True
                    break

            if not found:
                break

        if len(helix) >= 10:
            helices.append(helix)

    return helices


def smooth_fiber_spline(points, fiber_indices, n_samples=50):
    """Create smooth spline through fiber points (like protein backbone)."""
    fiber_points = points[fiber_indices]

    if len(fiber_points) < 4:
        return fiber_points

    try:
        # Fit cubic spline
        t = np.linspace(0, 1, len(fiber_points))
        t_smooth = np.linspace(0, 1, n_samples)

        # Spline for each coordinate
        smooth_points = np.zeros((n_samples, 3))
        for i in range(3):
            cs = CubicSpline(t, fiber_points[:, i])
            smooth_points[:, i] = cs(t_smooth)

        return smooth_points
    except:
        return fiber_points


def create_tube_mesh(path, radius=0.02, n_sides=8):
    """Create tube mesh around a path (like ribbon diagram)."""
    n_points = len(path)
    if n_points < 2:
        return None, None

    vertices = []
    faces = []

    for i in range(n_points):
        # Compute tangent
        if i == 0:
            tangent = path[1] - path[0]
        elif i == n_points - 1:
            tangent = path[-1] - path[-2]
        else:
            tangent = path[i + 1] - path[i - 1]

        tangent = tangent / (np.linalg.norm(tangent) + 1e-8)

        # Compute perpendicular vectors
        perp1 = np.cross(tangent, [1, 0, 0]) if abs(tangent[0]) < 0.9 else np.cross(tangent, [0, 1, 0])
        perp1 = perp1 / (np.linalg.norm(perp1) + 1e-8)
        perp2 = np.cross(tangent, perp1)

        # Create circle of vertices
        for j in range(n_sides):
            angle = 2 * np.pi * j / n_sides
            offset = radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
            vertices.append(path[i] + offset)

    vertices = np.array(vertices)

    # Create faces connecting adjacent circles
    for i in range(n_points - 1):
        for j in range(n_sides):
            v0 = i * n_sides + j
            v1 = i * n_sides + (j + 1) % n_sides
            v2 = (i + 1) * n_sides + (j + 1) % n_sides
            v3 = (i + 1) * n_sides + j

            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])

    return vertices, np.array(faces)


def compute_ambient_occlusion(points, n_samples=32, radius=0.3):
    """Compute ambient occlusion for depth visualization."""
    tree = KDTree(points)
    ao = np.zeros(len(points))

    for i, p in enumerate(points):
        # Count nearby points (more neighbors = more occluded)
        neighbors = tree.query_ball_point(p, radius)
        ao[i] = len(neighbors)

    # Normalize
    ao = ao / ao.max()
    return 1 - ao  # Invert: less neighbors = brighter


def render_fibration_matplotlib(points, fibers, helices, output_path):
    """Render fibration with protein-style visualization."""

    fig = plt.figure(figsize=(24, 18))

    # 1. Backbone trace (like protein backbone)
    ax = fig.add_subplot(2, 3, 1, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="gray", s=0.5, alpha=0.1)

    colors = plt.cm.hsv(np.linspace(0, 1, len(fibers)))
    for fiber, color in zip(fibers[:30], colors, strict=False):
        smooth_path = smooth_fiber_spline(points, fiber)
        ax.plot(
            smooth_path[:, 0],
            smooth_path[:, 1],
            smooth_path[:, 2],
            color=color,
            linewidth=1.5,
            alpha=0.8,
        )

    ax.set_title("Fiber Backbone (3-adic paths)")
    ax.view_init(elev=20, azim=45)

    # 2. Tube rendering (like ribbon diagram)
    ax = fig.add_subplot(2, 3, 2, projection="3d")

    for fiber, color in zip(fibers[:15], colors, strict=False):
        smooth_path = smooth_fiber_spline(points, fiber, n_samples=30)
        verts, faces = create_tube_mesh(smooth_path, radius=0.015, n_sides=6)
        if verts is not None:
            mesh = Poly3DCollection(verts[faces], alpha=0.7, facecolor=color, edgecolor="none")
            ax.add_collection3d(mesh)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_title("Tube Rendering (Ribbon Style)")
    ax.view_init(elev=20, azim=45)

    # 3. Helical fibers (Hopf fibration)
    ax = fig.add_subplot(2, 3, 3, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="gray", s=0.5, alpha=0.05)

    helix_colors = plt.cm.plasma(np.linspace(0, 1, len(helices)))
    for helix, color in zip(helices[:25], helix_colors, strict=False):
        smooth_path = smooth_fiber_spline(points, helix)
        ax.plot(
            smooth_path[:, 0],
            smooth_path[:, 1],
            smooth_path[:, 2],
            color=color,
            linewidth=2,
            alpha=0.9,
        )

    ax.set_title("Helical Fibers (Hopf Structure)")
    ax.view_init(elev=20, azim=45)

    # 4. Ambient occlusion shading
    ax = fig.add_subplot(2, 3, 4, projection="3d")
    ao = compute_ambient_occlusion(points, radius=0.15)
    scatter = ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=ao,
        cmap="bone",
        s=2,
        alpha=0.8,
    )
    ax.set_title("Ambient Occlusion (Depth)")
    ax.view_init(elev=20, azim=45)
    plt.colorbar(scatter, ax=ax, shrink=0.5, label="Exposure")

    # 5. Multi-layer transparency (showing internal structure)
    ax = fig.add_subplot(2, 3, 5, projection="3d")

    # Sort by depth for proper alpha blending
    view_dir = np.array([1, 1, 1]) / np.sqrt(3)
    depths = points @ view_dir
    depth_order = np.argsort(depths)

    # Color by depth layer
    n_layers = 5
    layer_size = len(points) // n_layers

    for layer in range(n_layers):
        start = layer * layer_size
        end = (layer + 1) * layer_size if layer < n_layers - 1 else len(points)
        layer_indices = depth_order[start:end]

        alpha = 0.3 + 0.5 * (layer / n_layers)  # Back layers more transparent
        ax.scatter(
            points[layer_indices, 0],
            points[layer_indices, 1],
            points[layer_indices, 2],
            c=plt.cm.viridis(layer / n_layers),
            s=2,
            alpha=alpha,
        )

    ax.set_title("Multi-Layer Transparency")
    ax.view_init(elev=20, azim=45)

    # 6. Combined fibration view
    ax = fig.add_subplot(2, 3, 6, projection="3d")

    # Background points with AO
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=ao,
        cmap="gray_r",
        s=1,
        alpha=0.2,
    )

    # Overlay key fibers
    for fiber, color in zip(fibers[:10], colors, strict=False):
        smooth_path = smooth_fiber_spline(points, fiber)
        ax.plot(
            smooth_path[:, 0],
            smooth_path[:, 1],
            smooth_path[:, 2],
            color=color,
            linewidth=2.5,
            alpha=0.9,
        )

    # Overlay helices with different style
    for helix, color in zip(helices[:5], helix_colors, strict=False):
        smooth_path = smooth_fiber_spline(points, helix)
        ax.plot(
            smooth_path[:, 0],
            smooth_path[:, 1],
            smooth_path[:, 2],
            color="white",
            linewidth=1,
            alpha=0.5,
            linestyle="--",
        )

    ax.set_title("Combined Fibration View")
    ax.view_init(elev=25, azim=60)

    plt.suptitle(
        "Calabi-Yau Internal Fibration Structure\n(Protein-Style Rendering)",
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path / "fibration_structure.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path / 'fibration_structure.png'}")


def export_fibration_data(points, fibers, helices, ao, output_path):
    """Export fibration data for Three.js rendering."""

    # Smooth all fibers
    smooth_fibers = []
    for fiber in fibers:
        smooth_path = smooth_fiber_spline(points, fiber, n_samples=30)
        smooth_fibers.append(smooth_path.tolist())

    smooth_helices = []
    for helix in helices:
        smooth_path = smooth_fiber_spline(points, helix, n_samples=30)
        smooth_helices.append(smooth_path.tolist())

    # Create tube meshes for key fibers
    tube_meshes = []
    for fiber in fibers[:30]:
        smooth_path = smooth_fiber_spline(points, fiber, n_samples=25)
        verts, faces = create_tube_mesh(smooth_path, radius=0.012, n_sides=6)
        if verts is not None:
            tube_meshes.append({"vertices": verts.tolist(), "faces": faces.tolist()})

    data = {
        "points": points.tolist(),
        "ambient_occlusion": ao.tolist(),
        "fibers": smooth_fibers,
        "helices": smooth_helices,
        "tube_meshes": tube_meshes,
        "n_points": len(points),
        "n_fibers": len(fibers),
        "n_helices": len(helices),
    }

    with open(output_path / "fibration_data.json", "w") as f:
        json.dump(data, f)
    print(f"Saved: {output_path / 'fibration_data.json'}")

    return data


def main():
    output_path = VIZ_DIR / "calabi_yau"
    output_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load embeddings
    print("\nLoading embeddings...")
    data = load_embeddings(str(CHECKPOINTS_DIR / "v5_5" / "latest.pt"), device)
    z_A = data["z_A"]

    # Project to 3D
    print("Projecting to 3D...")
    points = calabi_yau_projection(z_A)
    print(f"Points shape: {points.shape}")

    # Compute 3-adic neighbor structure
    print("\nComputing 3-adic neighbor graph...")
    neighbors = compute_3adic_neighbors()

    # Trace fibers
    print("Tracing fiber paths...")
    fibers = trace_fiber_paths(points, neighbors, n_fibers=100, fiber_length=80)
    print(f"Found {len(fibers)} fibers")

    # Trace helical structures
    print("Tracing helical fibers...")
    helices = trace_helical_fibers(points, n_helices=50, points_per_helix=80)
    print(f"Found {len(helices)} helices")

    # Compute ambient occlusion
    print("Computing ambient occlusion...")
    ao = compute_ambient_occlusion(points)

    # Render static visualization
    print("\nRendering fibration structure...")
    render_fibration_matplotlib(points, fibers, helices, output_path)

    # Export for Three.js
    print("\nExporting fibration data...")
    export_fibration_data(points, fibers, helices, ao, output_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
