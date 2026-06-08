# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""
Calabi-Yau Fibration Visualization from v5.8 Training Artifacts

Extracts multi-layer embeddings from v5.8 checkpoint to create 64D+ representation,
then projects to 3D Calabi-Yau manifold with fibration and intertwining structures.

Uses protein-style rendering for internal fibration visualization.
"""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import splev, splprep

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config.paths import CHECKPOINTS_DIR

# Output directory
OUTPUT_DIR = "outputs/viz/calabi_yau_v58"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


class MultiLayerExtractor(nn.Module):
    """Extract activations from multiple encoder layers."""

    def __init__(self, encoder_weights, layer_dims=None):
        if layer_dims is None:
            layer_dims = [256, 128, 64, 16]
        super().__init__()
        self.layers = nn.ModuleList()

        # Build encoder layers
        in_dim = 9
        for _i, out_dim in enumerate(layer_dims[:-1]):
            self.layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim

        # Final mu layer
        self.fc_mu = nn.Linear(layer_dims[-2], layer_dims[-1])

    def forward(self, x, return_all=True):
        """Forward pass returning all intermediate activations."""
        activations = []

        h = x
        for _i, layer in enumerate(self.layers):
            h = torch.relu(layer(h))
            activations.append(h)

        mu = self.fc_mu(h)
        activations.append(mu)

        if return_all:
            return activations
        return mu


def load_v58_checkpoint():
    """Load v5.8 checkpoint and extract model weights."""
    ckpt_path = str(CHECKPOINTS_DIR / "v5_8" / "latest.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    print(f"Loaded v5.8 checkpoint - Epoch {ckpt['epoch']}")
    print(f"Best coverage: {ckpt['best_coverage']:.2f}%")
    print(f"Best correlation: {ckpt['best_corr']:.4f}")

    return ckpt


def build_multi_layer_extractor(ckpt, vae="A"):
    """Build extractor with weights from checkpoint."""
    prefix = f"encoder_{vae}"
    state_dict = ckpt["model"]

    extractor = MultiLayerExtractor(state_dict)

    # Load weights
    extractor.layers[0].weight.data = state_dict[f"{prefix}.encoder.0.weight"]
    extractor.layers[0].bias.data = state_dict[f"{prefix}.encoder.0.bias"]
    extractor.layers[1].weight.data = state_dict[f"{prefix}.encoder.2.weight"]
    extractor.layers[1].bias.data = state_dict[f"{prefix}.encoder.2.bias"]
    extractor.layers[2].weight.data = state_dict[f"{prefix}.encoder.4.weight"]
    extractor.layers[2].bias.data = state_dict[f"{prefix}.encoder.4.bias"]
    extractor.fc_mu.weight.data = state_dict[f"{prefix}.fc_mu.weight"]
    extractor.fc_mu.bias.data = state_dict[f"{prefix}.fc_mu.bias"]

    return extractor.to(device).eval()


def generate_all_operations():
    """Generate all 3^9 = 19683 ternary operation vectors."""
    ops = []
    for i in range(19683):
        digits = []
        n = i
        for _ in range(9):
            digits.append(n % 3)
            n //= 3
        ops.append(digits)
    return torch.tensor(ops, dtype=torch.float32, device=device)


def extract_multilayer_embeddings(extractor_A, extractor_B, ops):
    """Extract embeddings from all layers of both VAEs.

    Creates 64D representation:
    - VAE-A: 256 + 128 + 64 + 16 = 464D (we'll sample key dimensions)
    - VAE-B: 256 + 128 + 64 + 16 = 464D

    For 64D we take: 16 (mu_A) + 16 (mu_B) + 16 (layer2_A) + 16 (layer2_B) = 64D
    """
    with torch.no_grad():
        acts_A = extractor_A(ops, return_all=True)
        acts_B = extractor_B(ops, return_all=True)

    # Layer activations: [256D, 128D, 64D, 16D]
    # Create 64D embedding by combining key representations
    layer2_A = acts_A[2]  # 64D
    layer2_B = acts_B[2]  # 64D
    mu_A = acts_A[3]  # 16D
    mu_B = acts_B[3]  # 16D

    # 64D = 16 + 16 + 16 + 16
    embedding_64d = torch.cat([mu_A, mu_B, layer2_A[:, :16], layer2_B[:, :16]], dim=1)

    # 128D for richer structure
    embedding_128d = torch.cat([mu_A, mu_B, layer2_A, layer2_B], dim=1)

    # Full representation (all layers concatenated)
    full_A = torch.cat(acts_A, dim=1)  # 256+128+64+16 = 464D
    full_B = torch.cat(acts_B, dim=1)  # 464D
    embedding_full = torch.cat([full_A, full_B], dim=1)  # 928D

    return {
        "32d": torch.cat([mu_A, mu_B], dim=1),
        "64d": embedding_64d,
        "128d": embedding_128d,
        "928d": embedding_full,
        "mu_A": mu_A,
        "mu_B": mu_B,
        "layer2_A": layer2_A,
        "layer2_B": layer2_B,
    }


# === Calabi-Yau Projection Methods ===


def calabi_yau_quintic_fibration(z, n_complex=5):
    """
    Project high-dimensional embedding to 3D via Calabi-Yau quintic fibration.

    Uses the quintic threefold: z1^5 + z2^5 + z3^5 + z4^5 + z5^5 = 0
    with fibration structure showing torus fibers over base.
    """
    z = z.cpu().numpy() if torch.is_tensor(z) else z
    n_points, dim = z.shape

    # Normalize to unit sphere
    z_norm = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)

    # Group dimensions into complex coordinates
    n_complex_pairs = dim // 2
    complex_coords = []
    for i in range(min(n_complex, n_complex_pairs)):
        re = z_norm[:, 2 * i]
        im = z_norm[:, 2 * i + 1] if 2 * i + 1 < dim else np.zeros_like(re)
        complex_coords.append(re + 1j * im)

    # Pad if needed
    while len(complex_coords) < n_complex:
        complex_coords.append(np.zeros(n_points, dtype=complex))

    # Quintic constraint: sum(z_i^5) = 0
    # Project onto the constraint surface
    z_powers = np.array([c**5 for c in complex_coords])
    constraint_violation = np.sum(z_powers, axis=0)

    # Fibration coordinates: base (B2) and fiber (T2)
    # Base coordinates from z1, z2
    base_x = np.real(complex_coords[0] * np.conj(complex_coords[1]))
    base_y = np.imag(complex_coords[0] * np.conj(complex_coords[1]))

    # Fiber coordinate from phase relationships
    phase_sum = np.angle(complex_coords[0]) + np.angle(complex_coords[2])
    fiber_z = np.sin(phase_sum * 2.5) * np.abs(constraint_violation) * 0.5

    # Add intertwining from higher-order terms
    if len(complex_coords) >= 4:
        intertwine = np.real(complex_coords[2] * complex_coords[3])
        fiber_z += 0.3 * intertwine

    return np.column_stack([base_x, base_y, fiber_z])


def hopf_fibration_intertwined(z):
    """
    Hopf fibration S^(2n-1) -> CP^(n-1) with intertwined fibers.

    Shows how fibers link and intertwine in the projection.
    """
    z = z.cpu().numpy() if torch.is_tensor(z) else z
    n_points, dim = z.shape

    # Normalize to sphere
    z_norm = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)

    # Create complex pairs
    n_pairs = dim // 2
    w = []
    for i in range(n_pairs):
        re = z_norm[:, 2 * i]
        im = z_norm[:, 2 * i + 1] if 2 * i + 1 < dim else np.zeros_like(re)
        w.append(re + 1j * im)

    # Hopf map: (w0, w1) -> (2*Re(w0*conj(w1)), 2*Im(w0*conj(w1)), |w0|^2 - |w1|^2)
    if len(w) >= 2:
        product = w[0] * np.conj(w[1])
        x = 2 * np.real(product)
        y = 2 * np.imag(product)
        z_coord = np.abs(w[0]) ** 2 - np.abs(w[1]) ** 2
    else:
        x = np.real(w[0])
        y = np.imag(w[0])
        z_coord = np.zeros_like(x)

    # Add intertwining from additional dimensions
    if len(w) >= 4:
        twist = np.angle(w[2]) - np.angle(w[3])
        x += 0.2 * np.cos(twist * 3)
        y += 0.2 * np.sin(twist * 3)
        z_coord += 0.15 * np.sin(twist * 2)

    # Additional layer intertwining
    if len(w) >= 6:
        twist2 = np.angle(w[4]) + np.angle(w[5])
        x += 0.1 * np.cos(twist2 * 4)
        y += 0.1 * np.sin(twist2 * 4)

    return np.column_stack([x, y, z_coord])


def k3_surface_projection(z):
    """
    Project to K3 surface (4D Calabi-Yau) then to 3D.

    K3 is a quartic surface in CP^3: w^4 + x^4 + y^4 + z^4 = 0
    """
    z = z.cpu().numpy() if torch.is_tensor(z) else z
    n_points, dim = z.shape

    # Normalize
    z_norm = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)

    # Group into 4 complex coordinates for K3
    dim // 4
    groups = []
    for i in range(4):
        start = i * (dim // 4)
        end = start + (dim // 4)
        group_sum = np.sum(z_norm[:, start:end], axis=1)
        groups.append(group_sum)

    # K3 quartic constraint
    quartic_sum = sum(g**4 for g in groups)

    # Project to 3D preserving K3 structure
    x = groups[0] * groups[1] - groups[2] * groups[3]
    y = groups[0] * groups[2] + groups[1] * groups[3]
    z_coord = groups[0] * groups[3] - groups[1] * groups[2]

    # Add surface curvature from quartic
    curvature = np.sign(quartic_sum) * np.abs(quartic_sum) ** 0.25
    z_coord += 0.2 * curvature

    return np.column_stack([x, y, z_coord])


def mirror_symmetric_fibration(z):
    """
    Mirror symmetry projection showing dual fibration structure.

    Calabi-Yau manifolds come in mirror pairs with exchanged Hodge numbers.
    This projection shows both the original and mirror structure intertwined.
    """
    z = z.cpu().numpy() if torch.is_tensor(z) else z
    n_points, dim = z.shape

    # Split into two halves for mirror pair
    half = dim // 2
    z_original = z[:, :half]
    z_mirror = z[:, half:]

    # Normalize each half
    z_orig_norm = z_original / (np.linalg.norm(z_original, axis=1, keepdims=True) + 1e-8)
    z_mirr_norm = z_mirror / (np.linalg.norm(z_mirror, axis=1, keepdims=True) + 1e-8)

    # Original fibration (complex structure)
    orig_x = np.sum(z_orig_norm[:, ::2] * z_orig_norm[:, 1::2], axis=1)
    orig_y = np.sum(z_orig_norm[:, ::2] ** 2 - z_orig_norm[:, 1::2] ** 2, axis=1)

    # Mirror fibration (symplectic structure)
    mirr_x = np.sum(z_mirr_norm[:, ::2] * z_mirr_norm[:, 1::2], axis=1)
    mirr_z = np.sum(z_mirr_norm[:, ::2] ** 2 - z_mirr_norm[:, 1::2] ** 2, axis=1)

    # Intertwine: x from original, y interpolated, z from mirror
    x = orig_x
    y = 0.5 * (orig_y + mirr_x)  # Interpolated
    z_coord = mirr_z

    # Add linking number contribution
    phase_orig = np.arctan2(orig_y, orig_x)
    phase_mirr = np.arctan2(mirr_z, mirr_x)
    linking = np.sin(phase_orig - phase_mirr)
    z_coord += 0.2 * linking

    return np.column_stack([x, y, z_coord])


# === Fibration Structure Analysis ===


def compute_3adic_distance(i, j):
    """Compute 3-adic distance between two operation indices."""
    diff = abs(i - j)
    if diff == 0:
        return float("inf")

    # Count factors of 3
    v3 = 0
    while diff % 3 == 0:
        v3 += 1
        diff //= 3

    return 3 ** (-v3)


def build_3adic_neighbor_graph(n_ops=19683, k_neighbors=6):
    """Build neighbor graph based on 3-adic distance."""
    print("Building 3-adic neighbor graph...")

    neighbors = []
    for i in range(n_ops):
        # Find k nearest 3-adic neighbors
        distances = []
        for j in range(n_ops):
            if i != j:
                d = compute_3adic_distance(i, j)
                distances.append((d, j))

        distances.sort(key=lambda x: x[0])
        neighbors.append([idx for _, idx in distances[:k_neighbors]])

    return neighbors


def trace_fibration_paths(points, neighbors, n_fibers=100, fiber_length=50):
    """Trace fiber paths through the manifold following 3-adic structure."""
    print(f"Tracing {n_fibers} fibration paths...")

    n_points = len(points)
    fibers = []

    # Start fibers at evenly distributed points
    start_indices = np.linspace(0, n_points - 1, n_fibers, dtype=int)

    for start_idx in start_indices:
        fiber = [start_idx]
        current = start_idx
        visited = {start_idx}

        for _ in range(fiber_length - 1):
            # Choose next point from neighbors not yet visited
            candidates = [n for n in neighbors[current] if n not in visited]

            if not candidates:
                # Backtrack or use any neighbor
                candidates = neighbors[current]

            if candidates:
                # Choose neighbor that continues fiber direction
                if len(fiber) >= 2:
                    prev_dir = points[fiber[-1]] - points[fiber[-2]]
                    best_idx = None
                    best_alignment = -float("inf")
                    for c in candidates:
                        new_dir = points[c] - points[current]
                        alignment = np.dot(prev_dir, new_dir)
                        if alignment > best_alignment:
                            best_alignment = alignment
                            best_idx = c
                    next_idx = best_idx if best_idx is not None else candidates[0]
                else:
                    next_idx = candidates[0]

                fiber.append(next_idx)
                visited.add(next_idx)
                current = next_idx
            else:
                break

        fibers.append(fiber)

    return fibers


def compute_intertwining_number(fiber1, fiber2, points):
    """Compute linking/intertwining number between two fibers."""
    # Simplified: count number of times fibers cross in 2D projections
    p1 = points[fiber1]
    p2 = points[fiber2]

    crossings = 0
    for i in range(len(p1) - 1):
        for j in range(len(p2) - 1):
            # Check if segments cross in xy plane
            a1, a2 = p1[i, :2], p1[i + 1, :2]
            b1, b2 = p2[j, :2], p2[j + 1, :2]

            # Simplified crossing check
            d1 = np.cross(b2 - b1, a1 - b1)
            d2 = np.cross(b2 - b1, a2 - b1)
            d3 = np.cross(a2 - a1, b1 - a1)
            d4 = np.cross(a2 - a1, b2 - a1)

            if d1 * d2 < 0 and d3 * d4 < 0:
                # Determine over/under from z coordinate
                t = np.cross(b1 - a1, b2 - b1) / (np.cross(a2 - a1, b2 - b1) + 1e-10)
                z_a = p1[i, 2] + t * (p1[i + 1, 2] - p1[i, 2])
                z_b = p2[j, 2] + t * (p2[j + 1, 2] - p2[j, 2])

                crossings += 1 if z_a > z_b else -1

    return crossings


def smooth_fiber_spline(points, fiber_indices, n_samples=100):
    """Create smooth spline through fiber points."""
    fiber_points = points[fiber_indices]

    if len(fiber_points) < 4:
        return fiber_points

    try:
        tck, u = splprep(
            [fiber_points[:, 0], fiber_points[:, 1], fiber_points[:, 2]],
            s=0.1,
            k=min(3, len(fiber_points) - 1),
        )
        u_new = np.linspace(0, 1, n_samples)
        smooth = np.array(splev(u_new, tck)).T
        return smooth
    except:
        return fiber_points


def create_tube_mesh(path, radius=0.015, n_sides=8):
    """Create tube mesh around path (protein ribbon style)."""
    n_points = len(path)

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

        # Create perpendicular vectors
        perp1 = np.cross(tangent, [1, 0, 0]) if abs(tangent[0]) < 0.9 else np.cross(tangent, [0, 1, 0])
        perp1 = perp1 / (np.linalg.norm(perp1) + 1e-8)
        perp2 = np.cross(tangent, perp1)

        # Create ring of vertices
        for j in range(n_sides):
            angle = 2 * np.pi * j / n_sides
            offset = radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
            vertices.append(path[i] + offset)

        # Create faces connecting to previous ring
        if i > 0:
            for j in range(n_sides):
                v1 = (i - 1) * n_sides + j
                v2 = (i - 1) * n_sides + (j + 1) % n_sides
                v3 = i * n_sides + (j + 1) % n_sides
                v4 = i * n_sides + j
                faces.append([v1, v2, v3])
                faces.append([v1, v3, v4])

    return np.array(vertices), np.array(faces)


# === Visualization ===


def render_fibration_structure(points, fibers, projection_name, output_path):
    """Render fibration structure with protein-style visualization."""
    fig = plt.figure(figsize=(16, 12))

    # Main 3D plot
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")

    # Plot points with color by position
    colors = np.arctan2(points[:, 1], points[:, 0])
    ax1.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=colors,
        cmap="twilight",
        s=1,
        alpha=0.3,
    )

    # Plot fiber backbones
    cmap = plt.cm.viridis
    for i, fiber in enumerate(fibers[:30]):  # Show first 30 fibers
        smooth_path = smooth_fiber_spline(points, fiber)
        color = cmap(i / 30)
        ax1.plot(
            smooth_path[:, 0],
            smooth_path[:, 1],
            smooth_path[:, 2],
            color=color,
            linewidth=1.5,
            alpha=0.8,
        )

    ax1.set_title(f"{projection_name}\nFibration Backbone Structure", fontsize=12)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # Intertwining detail view
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")

    # Zoom into region with intertwining
    center = np.mean(points, axis=0)
    mask = np.linalg.norm(points - center, axis=1) < 0.5
    central_points = points[mask]

    ax2.scatter(
        central_points[:, 0],
        central_points[:, 1],
        central_points[:, 2],
        c="blue",
        s=3,
        alpha=0.4,
    )

    # Show fiber tubes in central region
    for i, fiber in enumerate(fibers[:10]):
        smooth_path = smooth_fiber_spline(points, fiber)
        # Filter to central region
        central_mask = np.linalg.norm(smooth_path - center, axis=1) < 0.6
        if np.any(central_mask):
            central_path = smooth_path[central_mask]
            if len(central_path) > 3:
                ax2.plot(
                    central_path[:, 0],
                    central_path[:, 1],
                    central_path[:, 2],
                    linewidth=3,
                    alpha=0.9,
                    color=plt.cm.Set1(i / 10),
                )

    ax2.set_title("Intertwining Detail (Central Region)", fontsize=12)

    # 2D projections showing linking
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(
        points[:, 0],
        points[:, 1],
        c=points[:, 2],
        cmap="viridis",
        s=1,
        alpha=0.3,
    )
    for i, fiber in enumerate(fibers[:20]):
        smooth_path = smooth_fiber_spline(points, fiber)
        ax3.plot(smooth_path[:, 0], smooth_path[:, 1], linewidth=1, alpha=0.7)
    ax3.set_title("XY Projection (Z as color)", fontsize=12)
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_aspect("equal")

    # Fiber density plot
    ax4 = fig.add_subplot(2, 2, 4)

    # Compute fiber density
    from scipy.stats import gaussian_kde

    if len(points) > 100:
        xy = points[:, :2].T
        kde = gaussian_kde(xy)
        xi, yi = np.mgrid[
            points[:, 0].min() : points[:, 0].max() : 100j,
            points[:, 1].min() : points[:, 1].max() : 100j,
        ]
        zi = kde(np.vstack([xi.flatten(), yi.flatten()]))
        ax4.contourf(xi, yi, zi.reshape(xi.shape), levels=20, cmap="hot")
        ax4.set_title("Fiber Density Distribution", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def export_fibration_data(points, fibers, projection_name, output_path):
    """Export fibration data for Three.js visualization."""

    # Smooth all fibers
    smooth_fibers = []
    tube_meshes = []

    for fiber in fibers:
        smooth_path = smooth_fiber_spline(points, fiber, n_samples=50)
        smooth_fibers.append(smooth_path.tolist())

        # Create tube mesh
        vertices, faces = create_tube_mesh(smooth_path, radius=0.012)
        tube_meshes.append({"vertices": vertices.tolist(), "faces": faces.tolist()})

    # Compute intertwining matrix for first N fibers
    n_compute = min(20, len(fibers))
    intertwining_matrix = []
    for i in range(n_compute):
        row = []
        for j in range(n_compute):
            if i < j:
                linking = compute_intertwining_number(fibers[i], fibers[j], points)
                row.append(linking)
            elif i > j:
                row.append(-intertwining_matrix[j][i])
            else:
                row.append(0)
        intertwining_matrix.append(row)

    data = {
        "projection": projection_name,
        "n_points": len(points),
        "n_fibers": len(fibers),
        "points": points.tolist(),
        "fiber_indices": [list(f) for f in fibers],
        "smooth_fibers": smooth_fibers,
        "tube_meshes": tube_meshes,
        "intertwining_matrix": intertwining_matrix,
        "bounds": {
            "min": points.min(axis=0).tolist(),
            "max": points.max(axis=0).tolist(),
        },
    }

    with open(output_path, "w") as f:
        json.dump(data, f)

    print(f"Saved: {output_path}")
    return data


def main():
    # Load v5.8 checkpoint
    ckpt = load_v58_checkpoint()

    # Build multi-layer extractors
    print("\nBuilding multi-layer extractors...")
    extractor_A = build_multi_layer_extractor(ckpt, "A")
    extractor_B = build_multi_layer_extractor(ckpt, "B")

    # Generate all operations
    print("\nGenerating all 19683 ternary operations...")
    ops = generate_all_operations()

    # Extract multi-layer embeddings
    print("\nExtracting multi-layer embeddings...")
    embeddings = extract_multilayer_embeddings(extractor_A, extractor_B, ops)

    for name, emb in embeddings.items():
        if "layer" not in name and "mu" not in name:
            print(f"  {name}: {emb.shape}")

    # Build 3-adic neighbor graph (do once, reuse)
    neighbors = build_3adic_neighbor_graph(n_ops=19683, k_neighbors=6)

    # Process each embedding dimension
    projections = {
        "64d": {
            "embedding": embeddings["64d"],
            "methods": [
                ("Quintic Fibration", calabi_yau_quintic_fibration),
                ("Hopf Intertwined", hopf_fibration_intertwined),
                ("Mirror Symmetric", mirror_symmetric_fibration),
            ],
        },
        "128d": {
            "embedding": embeddings["128d"],
            "methods": [
                ("Quintic Fibration 128D", calabi_yau_quintic_fibration),
                ("K3 Surface", k3_surface_projection),
            ],
        },
    }

    all_results = {}

    for dim_name, config in projections.items():
        print(f"\n{'=' * 60}")
        print(f"Processing {dim_name} embeddings")
        print("=" * 60)

        emb = config["embedding"]

        for proj_name, proj_func in config["methods"]:
            print(f"\n--- {proj_name} ---")

            # Project to 3D
            points_3d = proj_func(emb)

            # Normalize for visualization
            points_3d = points_3d - points_3d.mean(axis=0)
            scale = np.max(np.abs(points_3d))
            points_3d = points_3d / scale if scale > 0 else points_3d

            print(f"Points range: [{points_3d.min():.3f}, {points_3d.max():.3f}]")

            # Trace fibration paths
            fibers = trace_fibration_paths(points_3d, neighbors, n_fibers=100, fiber_length=40)
            print(f"Traced {len(fibers)} fibers")

            # Render visualization
            safe_name = proj_name.replace(" ", "_").lower()
            output_png = os.path.join(OUTPUT_DIR, f"{dim_name}_{safe_name}.png")
            render_fibration_structure(points_3d, fibers, f"{proj_name} ({dim_name})", output_png)

            # Export data for Three.js
            output_json = os.path.join(OUTPUT_DIR, f"{dim_name}_{safe_name}.json")
            export_fibration_data(points_3d, fibers, f"{proj_name} ({dim_name})", output_json)

            all_results[f"{dim_name}_{safe_name}"] = {
                "points": points_3d,
                "fibers": fibers,
            }

    # Create summary comparison plot
    print("\n" + "=" * 60)
    print("Creating summary comparison...")

    fig = plt.figure(figsize=(20, 12))

    plot_idx = 1
    for key, result in all_results.items():
        ax = fig.add_subplot(2, 3, plot_idx, projection="3d")

        points = result["points"]
        fibers = result["fibers"]

        # Color by z-coordinate
        colors = points[:, 2]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=colors,
            cmap="twilight",
            s=1,
            alpha=0.3,
        )

        # Plot first 20 fibers
        for i, fiber in enumerate(fibers[:20]):
            smooth_path = smooth_fiber_spline(points, fiber)
            ax.plot(
                smooth_path[:, 0],
                smooth_path[:, 1],
                smooth_path[:, 2],
                linewidth=1.5,
                alpha=0.8,
                color=plt.cm.viridis(i / 20),
            )

        ax.set_title(key.replace("_", " ").title(), fontsize=10)
        plot_idx += 1

        if plot_idx > 6:
            break

    plt.suptitle(
        "Calabi-Yau Fibration Projections from v5.8 Multi-Layer Embeddings",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "summary_comparison.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'summary_comparison.png')}")

    print("\n" + "=" * 60)
    print("Done! Generated files:")
    for f in os.listdir(OUTPUT_DIR):
        print(f"  {f}")


if __name__ == "__main__":
    main()
