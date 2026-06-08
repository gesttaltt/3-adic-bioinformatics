# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""
Fast Calabi-Yau Fibration from v5.8 Training Artifacts

Efficient version using vectorized operations and sampling.
"""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from projections import quintic_fibration
from scipy.interpolate import splev, splprep
from scipy.spatial import KDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config.paths import CHECKPOINTS_DIR

OUTPUT_DIR = "outputs/viz/calabi_yau_v58"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading v5.8 checkpoint...")
ckpt = torch.load(
    str(CHECKPOINTS_DIR / "v5_8" / "latest.pt"),
    map_location="cpu",
    weights_only=False,
)
print(f"Epoch {ckpt['epoch']}, Coverage: {ckpt['best_coverage']:.2f}%, Correlation: {ckpt['best_corr']:.4f}")

state = ckpt["model"]

# Generate all operations
print("\nGenerating 19683 ternary operations...")
ops = []
for i in range(19683):
    digits = []
    n = i
    for _ in range(9):
        digits.append(n % 3)
        n //= 3
    ops.append(digits)
ops = torch.tensor(ops, dtype=torch.float32)

# Extract multi-layer embeddings
print("Extracting multi-layer embeddings...")


def encoder_forward(ops, prefix):
    h = ops
    h = torch.relu(
        F.linear(
            h,
            state[f"{prefix}.encoder.0.weight"],
            state[f"{prefix}.encoder.0.bias"],
        )
    )
    layer1 = h.clone()
    h = torch.relu(
        F.linear(
            h,
            state[f"{prefix}.encoder.2.weight"],
            state[f"{prefix}.encoder.2.bias"],
        )
    )
    layer2 = h.clone()
    h = torch.relu(
        F.linear(
            h,
            state[f"{prefix}.encoder.4.weight"],
            state[f"{prefix}.encoder.4.bias"],
        )
    )
    layer3 = h.clone()
    mu = F.linear(h, state[f"{prefix}.fc_mu.weight"], state[f"{prefix}.fc_mu.bias"])
    return {"layer1": layer1, "layer2": layer2, "layer3": layer3, "mu": mu}


acts_A = encoder_forward(ops, "encoder_A")
acts_B = encoder_forward(ops, "encoder_B")

# Create embeddings of different dimensions
emb_32d = torch.cat([acts_A["mu"], acts_B["mu"]], dim=1)
emb_64d = torch.cat(
    [
        acts_A["mu"],
        acts_B["mu"],
        acts_A["layer3"][:, :16],
        acts_B["layer3"][:, :16],
    ],
    dim=1,
)
emb_128d = torch.cat([acts_A["mu"], acts_B["mu"], acts_A["layer3"], acts_B["layer3"]], dim=1)

print(f"32D: {emb_32d.shape}, 64D: {emb_64d.shape}, 128D: {emb_128d.shape}")


# === Calabi-Yau Projection Functions ===
# quintic_fibration imported from projections.py


def hopf_intertwined(z):
    """Hopf fibration with intertwined fiber structure."""
    z = z.numpy() if torch.is_tensor(z) else z
    z_norm = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)

    dim = z.shape[1]
    n_pairs = dim // 2
    w = [z_norm[:, 2 * i] + 1j * z_norm[:, 2 * i + 1] for i in range(min(8, n_pairs))]

    # Hopf map
    prod = w[0] * np.conj(w[1])
    x = 2 * np.real(prod)
    y = 2 * np.imag(prod)
    z_coord = np.abs(w[0]) ** 2 - np.abs(w[1]) ** 2

    # Intertwining from higher dimensions
    if len(w) >= 4:
        twist = np.angle(w[2]) - np.angle(w[3])
        x += 0.15 * np.cos(twist * 3)
        y += 0.15 * np.sin(twist * 3)
        z_coord += 0.1 * np.sin(twist * 2)

    if len(w) >= 6:
        twist2 = np.angle(w[4]) + np.angle(w[5])
        x += 0.08 * np.cos(twist2 * 4)
        y += 0.08 * np.sin(twist2 * 4)

    return np.column_stack([x, y, z_coord])


def k3_projection(z):
    """K3 surface (4D Calabi-Yau) projection."""
    z = z.numpy() if torch.is_tensor(z) else z
    z_norm = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)

    dim = z.shape[1]
    g = [np.sum(z_norm[:, i * (dim // 4) : (i + 1) * (dim // 4)], axis=1) for i in range(4)]

    # K3 quartic structure
    x = g[0] * g[1] - g[2] * g[3]
    y = g[0] * g[2] + g[1] * g[3]
    z_coord = g[0] * g[3] - g[1] * g[2]

    # Curvature from quartic constraint
    quartic = sum(gi**4 for gi in g)
    z_coord += 0.15 * np.sign(quartic) * np.abs(quartic) ** 0.25

    return np.column_stack([x, y, z_coord])


def mirror_fibration(z):
    """Mirror symmetry dual fibration."""
    z = z.numpy() if torch.is_tensor(z) else z
    z_norm = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)

    half = z.shape[1] // 2
    z_A = z_norm[:, :half]
    z_B = z_norm[:, half:]

    # Original fibration
    x = np.sum(z_A[:, ::2] * z_A[:, 1::2], axis=1)
    y_orig = np.sum(z_A[:, ::2] ** 2 - z_A[:, 1::2] ** 2, axis=1)

    # Mirror fibration
    y_mirr = np.sum(z_B[:, ::2] * z_B[:, 1::2], axis=1)
    z_coord = np.sum(z_B[:, ::2] ** 2 - z_B[:, 1::2] ** 2, axis=1)

    # Intertwine
    y = 0.5 * (y_orig + y_mirr)

    # Linking contribution
    phase_A = np.arctan2(y_orig, x)
    phase_B = np.arctan2(z_coord, y_mirr)
    z_coord += 0.15 * np.sin(phase_A - phase_B)

    return np.column_stack([x, y, z_coord])


# === Fast Fiber Tracing ===


def trace_fibers_kdtree(points, n_fibers=100, fiber_length=40):
    """Trace fibers using KDTree for fast neighbor lookup."""
    print(f"Tracing {n_fibers} fibers...")

    tree = KDTree(points)
    n_points = len(points)
    fibers = []

    # Start points distributed across manifold
    start_indices = np.random.choice(n_points, n_fibers, replace=False)

    for start_idx in start_indices:
        fiber = [start_idx]
        current = start_idx
        visited = {start_idx}

        for _ in range(fiber_length - 1):
            # Find nearest unvisited neighbors
            distances, indices = tree.query(points[current], k=10)

            candidates = [idx for idx in indices if idx not in visited]
            if not candidates:
                break

            # Choose neighbor that continues fiber direction
            if len(fiber) >= 2:
                prev_dir = points[fiber[-1]] - points[fiber[-2]]
                prev_dir = prev_dir / (np.linalg.norm(prev_dir) + 1e-8)

                best_idx = candidates[0]
                best_score = -float("inf")
                for c in candidates[:5]:
                    new_dir = points[c] - points[current]
                    new_dir = new_dir / (np.linalg.norm(new_dir) + 1e-8)
                    score = np.dot(prev_dir, new_dir)
                    if score > best_score:
                        best_score = score
                        best_idx = c
                next_idx = best_idx
            else:
                next_idx = candidates[0]

            fiber.append(next_idx)
            visited.add(next_idx)
            current = next_idx

        if len(fiber) >= 5:
            fibers.append(fiber)

    return fibers


def smooth_fiber(points, fiber_indices, n_samples=50):
    """Smooth fiber with spline interpolation."""
    pts = points[fiber_indices]
    if len(pts) < 4:
        return pts

    try:
        tck, u = splprep([pts[:, 0], pts[:, 1], pts[:, 2]], s=0.1, k=min(3, len(pts) - 1))
        u_new = np.linspace(0, 1, n_samples)
        return np.array(splev(u_new, tck)).T
    except:
        return pts


def create_tube_mesh(path, radius=0.012, n_sides=8):
    """Create tube mesh around fiber path."""
    n = len(path)
    vertices = []
    faces = []

    for i in range(n):
        # Tangent
        if i == 0:
            t = path[1] - path[0]
        elif i == n - 1:
            t = path[-1] - path[-2]
        else:
            t = path[i + 1] - path[i - 1]
        t = t / (np.linalg.norm(t) + 1e-8)

        # Perpendiculars
        p1 = np.cross(t, [1, 0, 0]) if abs(t[0]) < 0.9 else np.cross(t, [0, 1, 0])
        p1 = p1 / (np.linalg.norm(p1) + 1e-8)
        p2 = np.cross(t, p1)

        # Ring
        for j in range(n_sides):
            angle = 2 * np.pi * j / n_sides
            vertices.append(path[i] + radius * (np.cos(angle) * p1 + np.sin(angle) * p2))

        # Faces
        if i > 0:
            for j in range(n_sides):
                v1 = (i - 1) * n_sides + j
                v2 = (i - 1) * n_sides + (j + 1) % n_sides
                v3 = i * n_sides + (j + 1) % n_sides
                v4 = i * n_sides + j
                faces.append([v1, v2, v3])
                faces.append([v1, v3, v4])

    return np.array(vertices), np.array(faces)


# === Main Processing ===

print("\n" + "=" * 60)
print("Processing Calabi-Yau Fibration Projections")
print("=" * 60)

projections = [
    ("64D Quintic", emb_64d, quintic_fibration),
    ("64D Hopf", emb_64d, hopf_intertwined),
    ("128D K3", emb_128d, k3_projection),
    ("128D Mirror", emb_128d, mirror_fibration),
]

all_results = {}

for name, emb, proj_func in projections:
    print(f"\n--- {name} ---")

    # Project to 3D
    points = proj_func(emb)

    # Normalize
    points = points - points.mean(axis=0)
    scale = np.max(np.abs(points))
    points = points / scale if scale > 0 else points

    print(f"Range: [{points.min():.3f}, {points.max():.3f}]")

    # Trace fibers
    fibers = trace_fibers_kdtree(points, n_fibers=100, fiber_length=40)
    print(f"Traced {len(fibers)} fibers")

    all_results[name] = {"points": points, "fibers": fibers}

    # Export JSON for Three.js
    safe_name = name.lower().replace(" ", "_")

    smooth_fibers = [smooth_fiber(points, f).tolist() for f in fibers]
    tube_meshes = []
    for f in fibers[:50]:  # Limit tubes for file size
        sf = smooth_fiber(points, f)
        v, f_idx = create_tube_mesh(sf)
        tube_meshes.append(
            {
                "vertices": v.tolist(),
                "faces": [[int(x) for x in face] for face in f_idx.tolist()],
            }
        )

    data = {
        "name": name,
        "n_points": len(points),
        "n_fibers": len(fibers),
        "points": points.tolist(),
        "fiber_indices": [[int(i) for i in f] for f in fibers],
        "smooth_fibers": smooth_fibers,
        "tube_meshes": tube_meshes,
        "bounds": {
            "min": points.min(axis=0).tolist(),
            "max": points.max(axis=0).tolist(),
        },
    }

    json_path = os.path.join(OUTPUT_DIR, f"{safe_name}.json")
    with open(json_path, "w") as f:
        json.dump(data, f)
    print(f"Saved: {json_path}")


# === Create Visualization ===

print("\nCreating visualization...")

fig = plt.figure(figsize=(20, 15))

for idx, (name, result) in enumerate(all_results.items()):
    points = result["points"]
    fibers = result["fibers"]

    # Main 3D view
    ax = fig.add_subplot(2, 4, idx + 1, projection="3d")

    # Points colored by position
    colors = np.arctan2(points[:, 1], points[:, 0])
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=colors,
        cmap="twilight",
        s=0.5,
        alpha=0.2,
    )

    # Fiber ribbons
    for i, fiber in enumerate(fibers[:25]):
        sf = smooth_fiber(points, fiber)
        color = plt.cm.viridis(i / 25)
        ax.plot(sf[:, 0], sf[:, 1], sf[:, 2], color=color, linewidth=1.5, alpha=0.8)

    ax.set_title(name, fontsize=11)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # 2D projection
    ax2 = fig.add_subplot(2, 4, idx + 5)
    ax2.scatter(
        points[:, 0],
        points[:, 1],
        c=points[:, 2],
        cmap="viridis",
        s=0.3,
        alpha=0.3,
    )
    for i, fiber in enumerate(fibers[:15]):
        sf = smooth_fiber(points, fiber)
        ax2.plot(
            sf[:, 0],
            sf[:, 1],
            linewidth=1,
            alpha=0.7,
            color=plt.cm.plasma(i / 15),
        )
    ax2.set_title(f"{name} (XY proj)", fontsize=10)
    ax2.set_aspect("equal")

plt.suptitle(
    "v5.8 Multi-Layer Embeddings: Calabi-Yau Fibration Projections\n"
    f"Coverage: {ckpt['best_coverage']:.1f}%, Correlation: {ckpt['best_corr']:.3f}",
    fontsize=14,
)
plt.tight_layout()

out_path = os.path.join(OUTPUT_DIR, "fibration_summary.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out_path}")

# List output files
print("\n" + "=" * 60)
print("Generated files:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
    print(f"  {f}: {size / 1024:.1f} KB")
