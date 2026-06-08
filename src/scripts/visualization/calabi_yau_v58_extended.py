# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""
Extended Calabi-Yau Fibration from v5.8 - 500 fibers, 8 projections

Cross-references multiple projection methods for comprehensive manifold analysis.
"""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import splev, splprep
from scipy.spatial import KDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config.paths import CHECKPOINTS_DIR

# Canonical projection implementations available in projections.py
# This script uses specialized variants for extended analysis

OUTPUT_DIR = "outputs/viz/calabi_yau_v58"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_FIBERS = 500
FIBER_LENGTH = 50

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
    layer1 = h.clone()  # 256D
    h = torch.relu(
        F.linear(
            h,
            state[f"{prefix}.encoder.2.weight"],
            state[f"{prefix}.encoder.2.bias"],
        )
    )
    layer2 = h.clone()  # 128D
    h = torch.relu(
        F.linear(
            h,
            state[f"{prefix}.encoder.4.weight"],
            state[f"{prefix}.encoder.4.bias"],
        )
    )
    layer3 = h.clone()  # 64D
    mu = F.linear(h, state[f"{prefix}.fc_mu.weight"], state[f"{prefix}.fc_mu.bias"])  # 16D
    return {"layer1": layer1, "layer2": layer2, "layer3": layer3, "mu": mu}


acts_A = encoder_forward(ops, "encoder_A")
acts_B = encoder_forward(ops, "encoder_B")

# Create embeddings of different dimensions
embeddings = {
    "32d": torch.cat([acts_A["mu"], acts_B["mu"]], dim=1),
    "64d": torch.cat(
        [
            acts_A["mu"],
            acts_B["mu"],
            acts_A["layer3"][:, :16],
            acts_B["layer3"][:, :16],
        ],
        dim=1,
    ),
    "128d": torch.cat([acts_A["mu"], acts_B["mu"], acts_A["layer3"], acts_B["layer3"]], dim=1),
    "192d": torch.cat(
        [
            acts_A["mu"],
            acts_B["mu"],
            acts_A["layer3"],
            acts_B["layer3"],
            acts_A["layer2"][:, :32],
            acts_B["layer2"][:, :32],
        ],
        dim=1,
    ),
    "256d": torch.cat(
        [
            acts_A["layer3"],
            acts_B["layer3"],
            acts_A["layer2"],
            acts_B["layer2"],
        ],
        dim=1,
    ),
    "512d": torch.cat([acts_A["layer1"], acts_B["layer1"]], dim=1),
}

for name, emb in embeddings.items():
    print(f"  {name}: {emb.shape}")


# === Extended Calabi-Yau Projection Functions ===


def quintic_fibration(z):
    """Calabi-Yau quintic threefold: z1^5 + z2^5 + z3^5 + z4^5 + z5^5 = 0"""
    z = z.numpy() if torch.is_tensor(z) else z
    z_norm = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    dim = z.shape[1]
    n_complex = min(5, dim // 2)

    w = [z_norm[:, 2 * i] + 1j * z_norm[:, 2 * i + 1] for i in range(n_complex)]
    constraint = sum(c**5 for c in w)

    x = np.real(w[0] * np.conj(w[1]))
    y = np.imag(w[0] * np.conj(w[1]))
    phase = np.angle(w[0]) + np.angle(w[2]) if len(w) > 2 else np.angle(w[0])
    z_coord = np.sin(phase * 2.5) * np.abs(constraint) * 0.3
    if len(w) >= 4:
        z_coord += 0.2 * np.real(w[2] * w[3])
    return np.column_stack([x, y, z_coord])


def hopf_fibration(z):
    """Hopf fibration S^(2n-1) -> CP^(n-1)"""
    z = z.numpy() if torch.is_tensor(z) else z
    z_norm = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    dim = z.shape[1]
    n_pairs = min(8, dim // 2)
    w = [z_norm[:, 2 * i] + 1j * z_norm[:, 2 * i + 1] for i in range(n_pairs)]

    prod = w[0] * np.conj(w[1])
    x = 2 * np.real(prod)
    y = 2 * np.imag(prod)
    z_coord = np.abs(w[0]) ** 2 - np.abs(w[1]) ** 2

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


def k3_surface(z):
    """K3 surface (4D Calabi-Yau): w^4 + x^4 + y^4 + z^4 = 0"""
    z = z.numpy() if torch.is_tensor(z) else z
    z_norm = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    dim = z.shape[1]
    g = [np.sum(z_norm[:, i * (dim // 4) : (i + 1) * (dim // 4)], axis=1) for i in range(4)]

    x = g[0] * g[1] - g[2] * g[3]
    y = g[0] * g[2] + g[1] * g[3]
    z_coord = g[0] * g[3] - g[1] * g[2]
    quartic = sum(gi**4 for gi in g)
    z_coord += 0.15 * np.sign(quartic) * np.abs(quartic) ** 0.25
    return np.column_stack([x, y, z_coord])


def mirror_symmetry(z):
    """Mirror symmetry dual fibration"""
    z = z.numpy() if torch.is_tensor(z) else z
    z_norm = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    half = z.shape[1] // 2
    z_A, z_B = z_norm[:, :half], z_norm[:, half:]

    x = np.sum(z_A[:, ::2] * z_A[:, 1::2], axis=1)
    y_orig = np.sum(z_A[:, ::2] ** 2 - z_A[:, 1::2] ** 2, axis=1)
    y_mirr = np.sum(z_B[:, ::2] * z_B[:, 1::2], axis=1)
    z_coord = np.sum(z_B[:, ::2] ** 2 - z_B[:, 1::2] ** 2, axis=1)

    y = 0.5 * (y_orig + y_mirr)
    phase_A = np.arctan2(y_orig, x)
    phase_B = np.arctan2(z_coord, y_mirr)
    z_coord += 0.15 * np.sin(phase_A - phase_B)
    return np.column_stack([x, y, z_coord])


def fermat_surface(z, degree=4):
    """Fermat surface: x^n + y^n + z^n + w^n = 0"""
    z = z.numpy() if torch.is_tensor(z) else z
    z_norm = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    dim = z.shape[1]

    # Group into 4 sections
    sections = [z_norm[:, i * (dim // 4) : (i + 1) * (dim // 4)] for i in range(4)]
    s = [np.sum(sec, axis=1) for sec in sections]

    # Fermat constraint
    fermat = sum(si**degree for si in s)

    # Stereographic-like projection
    denom = 1 + s[3] ** 2 + 1e-8
    x = s[0] / denom
    y = s[1] / denom
    z_coord = s[2] / denom + 0.1 * np.sign(fermat) * np.abs(fermat) ** (1 / degree)
    return np.column_stack([x, y, z_coord])


def torus_fibration(z):
    """T^2 fibration over S^2 (elliptic fibration)"""
    z = z.numpy() if torch.is_tensor(z) else z
    z_norm = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    dim = z.shape[1]

    # Base S^2 from first components
    base_theta = np.arctan2(z_norm[:, 1], z_norm[:, 0])
    base_phi = np.arccos(np.clip(z_norm[:, 2], -1, 1))

    # Fiber T^2 from middle components
    fiber_u = np.sum(z_norm[:, dim // 4 : dim // 2], axis=1) * 2 * np.pi
    fiber_v = np.sum(z_norm[:, dim // 2 : 3 * dim // 4], axis=1) * 2 * np.pi

    # Major and minor radii
    R, r = 1.0, 0.3

    # Torus coordinates modulated by base
    x = (R + r * np.cos(fiber_v)) * np.cos(fiber_u + base_theta)
    y = (R + r * np.cos(fiber_v)) * np.sin(fiber_u + base_theta)
    z_coord = r * np.sin(fiber_v) + 0.2 * np.cos(base_phi)
    return np.column_stack([x, y, z_coord])


def conifold_transition(z):
    """Conifold transition: xy - zw = t (singular at t=0)"""
    z = z.numpy() if torch.is_tensor(z) else z
    z_norm = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    dim = z.shape[1]

    # Create 4 complex coordinates
    n_complex = min(4, dim // 2)
    w = [z_norm[:, 2 * i] + 1j * z_norm[:, 2 * i + 1] for i in range(n_complex)]
    while len(w) < 4:
        w.append(np.zeros_like(w[0]))

    # Conifold equation: w0*w1 - w2*w3 = t
    t_param = np.sum(z_norm[:, -dim // 8 :], axis=1) * 0.1  # Deformation parameter
    conifold = w[0] * w[1] - w[2] * w[3] - t_param

    x = np.real(w[0] + w[1])
    y = np.imag(w[0] - w[1])
    z_coord = np.real(conifold) + 0.3 * np.imag(w[2] * np.conj(w[3]))
    return np.column_stack([x, y, z_coord])


def calabi_yau_3fold(z):
    """Generic CY3 projection using Kähler moduli"""
    z = z.numpy() if torch.is_tensor(z) else z
    z_norm = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    dim = z.shape[1]

    # Split into Kähler and complex structure moduli
    kahler = z_norm[:, : dim // 2]
    complex_struct = z_norm[:, dim // 2 :]

    # Kähler form contribution
    J = np.sum(kahler[:, ::2] * kahler[:, 1::2], axis=1)

    # Holomorphic 3-form contribution
    omega_re = np.sum(complex_struct[:, : complex_struct.shape[1] // 2], axis=1)
    omega_im = np.sum(complex_struct[:, complex_struct.shape[1] // 2 :], axis=1)

    x = J * np.cos(omega_re * np.pi)
    y = J * np.sin(omega_re * np.pi)
    z_coord = omega_im * np.sqrt(np.abs(J) + 0.1)
    return np.column_stack([x, y, z_coord])


# === Fiber Tracing ===


def trace_fibers_kdtree(points, n_fibers=N_FIBERS, fiber_length=FIBER_LENGTH):
    """Trace fibers using KDTree for fast neighbor lookup."""
    print(f"  Tracing {n_fibers} fibers (length {fiber_length})...")

    tree = KDTree(points)
    n_points = len(points)
    fibers = []

    # Distributed start points
    start_indices = np.random.choice(n_points, min(n_fibers, n_points), replace=False)

    for start_idx in start_indices:
        fiber = [start_idx]
        current = start_idx
        visited = {start_idx}

        for _ in range(fiber_length - 1):
            distances, indices = tree.query(points[current], k=12)
            candidates = [idx for idx in indices if idx not in visited]
            if not candidates:
                break

            if len(fiber) >= 2:
                prev_dir = points[fiber[-1]] - points[fiber[-2]]
                prev_dir = prev_dir / (np.linalg.norm(prev_dir) + 1e-8)

                best_idx = candidates[0]
                best_score = -float("inf")
                for c in candidates[:6]:
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


def smooth_fiber(points, fiber_indices, n_samples=60):
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


def create_tube_mesh(path, radius=0.008, n_sides=6):
    """Create thin tube mesh around fiber path."""
    n = len(path)
    vertices = []
    faces = []

    for i in range(n):
        if i == 0:
            t = path[1] - path[0]
        elif i == n - 1:
            t = path[-1] - path[-2]
        else:
            t = path[i + 1] - path[i - 1]
        t = t / (np.linalg.norm(t) + 1e-8)

        p1 = np.cross(t, [1, 0, 0]) if abs(t[0]) < 0.9 else np.cross(t, [0, 1, 0])
        p1 = p1 / (np.linalg.norm(p1) + 1e-8)
        p2 = np.cross(t, p1)

        for j in range(n_sides):
            angle = 2 * np.pi * j / n_sides
            vertices.append(path[i] + radius * (np.cos(angle) * p1 + np.sin(angle) * p2))

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

print("\n" + "=" * 70)
print(f"Processing Extended Calabi-Yau Fibration Projections ({N_FIBERS} fibers)")
print("=" * 70)

projections = [
    # (name, embedding_key, projection_function)
    ("Quintic 64D", "64d", quintic_fibration),
    ("Hopf 64D", "64d", hopf_fibration),
    ("K3 128D", "128d", k3_surface),
    ("Mirror 128D", "128d", mirror_symmetry),
    ("Fermat 192D", "192d", fermat_surface),
    ("Torus 192D", "192d", torus_fibration),
    ("Conifold 256D", "256d", conifold_transition),
    ("CY3 512D", "512d", calabi_yau_3fold),
]

all_results = {}

for name, emb_key, proj_func in projections:
    print(f"\n--- {name} ---")
    emb = embeddings[emb_key]

    # Project to 3D
    points = proj_func(emb)

    # Normalize
    points = points - points.mean(axis=0)
    scale = np.percentile(np.abs(points), 99)
    points = points / scale if scale > 0 else points

    print(f"  Range: [{points.min():.3f}, {points.max():.3f}]")

    # Trace fibers
    fibers = trace_fibers_kdtree(points)
    print(f"  Traced {len(fibers)} fibers")

    all_results[name] = {
        "points": points,
        "fibers": fibers,
        "emb_dim": emb_key,
    }

    # Export JSON
    safe_name = name.lower().replace(" ", "_")

    smooth_fibers = [smooth_fiber(points, f).tolist() for f in fibers]

    # Tube meshes for first 100 fibers only (file size)
    tube_meshes = []
    for f in fibers[:100]:
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
        "embedding_dim": emb_key,
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
    print(f"  Saved: {json_path}")


# === Create Visualization Grid ===

print("\nCreating visualization grid...")

fig = plt.figure(figsize=(24, 18))

for idx, (name, result) in enumerate(all_results.items()):
    points = result["points"]
    fibers = result["fibers"]
    emb_dim = result["emb_dim"]

    # 3D view
    ax = fig.add_subplot(2, 4, idx + 1, projection="3d")

    # Points colored by angle
    colors = np.arctan2(points[:, 1], points[:, 0])
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=colors,
        cmap="twilight",
        s=0.3,
        alpha=0.15,
    )

    # Show more fibers (every 5th for visibility)
    for i, fiber in enumerate(fibers[::5][:60]):
        sf = smooth_fiber(points, fiber)
        color = plt.cm.viridis(i / 60)
        ax.plot(sf[:, 0], sf[:, 1], sf[:, 2], color=color, linewidth=1.2, alpha=0.7)

    ax.set_title(f"{name}\n({len(fibers)} fibers, {emb_dim})", fontsize=10)
    ax.set_xlabel("X", fontsize=8)
    ax.set_ylabel("Y", fontsize=8)
    ax.set_zlabel("Z", fontsize=8)

plt.suptitle(
    f"v5.8 Multi-Layer Embeddings: 8 Calabi-Yau Projections ({N_FIBERS} fibers each)\n"
    f"Coverage: {ckpt['best_coverage']:.1f}%, Correlation: {ckpt['best_corr']:.3f}",
    fontsize=14,
)
plt.tight_layout()

out_path = os.path.join(OUTPUT_DIR, "extended_fibration_grid.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")


# === Cross-correlation analysis ===

print("\nComputing cross-projection correlations...")

proj_names = list(all_results.keys())
n_proj = len(proj_names)
corr_matrix = np.zeros((n_proj, n_proj))

for i, name_i in enumerate(proj_names):
    for j, name_j in enumerate(proj_names):
        pts_i = all_results[name_i]["points"]
        pts_j = all_results[name_j]["points"]

        # Correlation of distances from origin
        dist_i = np.linalg.norm(pts_i, axis=1)
        dist_j = np.linalg.norm(pts_j, axis=1)
        corr_matrix[i, j] = np.corrcoef(dist_i, dist_j)[0, 1]

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr_matrix, cmap="RdYlBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(n_proj))
ax.set_yticks(range(n_proj))
ax.set_xticklabels([n.split()[0] for n in proj_names], rotation=45, ha="right")
ax.set_yticklabels([n.split()[0] for n in proj_names])

for i in range(n_proj):
    for j in range(n_proj):
        ax.text(
            j,
            i,
            f"{corr_matrix[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=9,
        )

plt.colorbar(im, label="Distance Correlation")
plt.title("Cross-Projection Correlation Matrix")
plt.tight_layout()

corr_path = os.path.join(OUTPUT_DIR, "projection_correlation.png")
plt.savefig(corr_path, dpi=150)
plt.close()
print(f"Saved: {corr_path}")


# === Summary ===

print("\n" + "=" * 70)
print("Generated files:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
    print(f"  {f}: {size:.1f} KB")
print("=" * 70)
