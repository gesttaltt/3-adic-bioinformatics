# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Calabi-Yau Manifold Projections of VAE Latent Spaces.

Projects the 16D VAE latent embeddings onto 3D using Calabi-Yau inspired
geometric transformations. Calabi-Yau manifolds are Ricci-flat Kähler
manifolds that appear in string theory as compactification spaces.

Projection methods:
1. Quintic hypersurface slicing - z1^5 + z2^5 + z3^5 + z4^5 + z5^5 = 0
2. Complex coordinate pairing - treating R^16 as C^8
3. Hopf fibration inspired - S^15 -> S^7 -> S^3 projections
4. Holomorphic sectioning - real slices through complex structure
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.ternary_vae_v5_6 import DualNeuralVAEV5

from src.config.paths import VIZ_DIR
from src.data.generation import generate_all_ternary_operations


def load_embeddings(checkpoint_path, device="cuda"):
    """Load model and extract all 19683 latent embeddings."""
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

        # Also get reconstruction accuracy for coloring
        logits_A = model.decoder_A(mu_A)
        logits_B = model.decoder_B(mu_B)
        preds_A = logits_A.argmax(dim=-1) - 1
        preds_B = logits_B.argmax(dim=-1) - 1
        acc_A = (preds_A == x).float().mean(dim=1)
        acc_B = (preds_B == x).float().mean(dim=1)

    return {
        "z_A": mu_A.cpu().numpy(),
        "z_B": mu_B.cpu().numpy(),
        "var_A": logvar_A.exp().cpu().numpy(),
        "var_B": logvar_B.exp().cpu().numpy(),
        "acc_A": acc_A.cpu().numpy(),
        "acc_B": acc_B.cpu().numpy(),
        "operations": operations,
        "epoch": checkpoint.get("epoch", 0),
    }


def normalize_to_sphere(z, radius=1.0):
    """Normalize points to lie on a hypersphere."""
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)  # Avoid division by zero
    return z / norms * radius


def calabi_yau_quintic_projection(z, phase=0.0):
    """Project 16D points using quintic Calabi-Yau parametrization.

    The quintic threefold is defined by: sum(z_i^5) = 0 in CP^4
    We use a parametric approach inspired by this structure.

    Args:
        z: (N, 16) array of latent points
        phase: rotation phase for animation

    Returns:
        (N, 3) array of projected points
    """
    z.shape[0]

    # Pair dimensions as complex numbers: R^16 -> C^8
    z_complex = z[:, ::2] + 1j * z[:, 1::2]  # (N, 8) complex

    # Apply quintic-inspired transformation
    # Use first 5 complex dimensions for the quintic structure
    z5 = z_complex[:, :5]

    # Quintic surface parametrization with phase rotation
    theta = np.angle(z5) + phase
    r = np.abs(z5)

    # Map to 3D using quintic harmonics
    x = np.sum(r * np.cos(5 * theta), axis=1)
    y = np.sum(r * np.sin(5 * theta), axis=1)
    z_out = np.sum(r * np.cos(3 * theta) * np.sin(2 * theta), axis=1)

    # Normalize for visualization
    result = np.stack([x, y, z_out], axis=1)
    max_val = np.abs(result).max()
    if max_val > 0:
        result = result / max_val

    return result


def calabi_yau_fermat_projection(z, k=5):
    """Fermat quintic projection: x^k + y^k + z^k = 1 inspired.

    Creates a more structured Calabi-Yau visualization using
    the Fermat surface parametrization.
    """
    z.shape[0]

    # Use first 6 dimensions for parametrization
    u = z[:, 0:2]  # First complex coordinate
    v = z[:, 2:4]  # Second complex coordinate
    w = z[:, 4:6]  # Third complex coordinate

    # Convert to complex
    u_c = u[:, 0] + 1j * u[:, 1]
    v_c = v[:, 0] + 1j * v[:, 1]
    w_c = w[:, 0] + 1j * w[:, 1]

    # Fermat-like transformation
    # The real parts of z^(1/k) for different roots
    np.exp(2j * np.pi * np.arange(k) / k)

    x = np.real(np.abs(u_c) ** (2 / k) * np.exp(1j * np.angle(u_c) / k))
    y = np.real(np.abs(v_c) ** (2 / k) * np.exp(1j * np.angle(v_c) / k))
    z_out = np.real(np.abs(w_c) ** (2 / k) * np.exp(1j * np.angle(w_c) / k))

    # Add contribution from higher dimensions for richness
    for i in range(3, 8):
        x += 0.1 * z[:, 2 * i] * np.cos(i * np.pi / 8)
        y += 0.1 * z[:, 2 * i] * np.sin(i * np.pi / 8)
        z_out += 0.1 * z[:, 2 * i + 1] * np.cos(i * np.pi / 4)

    result = np.stack([x, y, z_out], axis=1)
    # Normalize
    max_val = np.percentile(np.abs(result), 99)
    if max_val > 0:
        result = result / max_val

    return result


def hopf_fibration_projection(z):
    """Project using Hopf fibration structure: S^15 -> S^7 -> S^3.

    The Hopf fibration is a beautiful geometric structure that
    maps higher-dimensional spheres to lower ones.
    """
    # First normalize to S^15
    z_sphere = normalize_to_sphere(z)

    # S^15 -> S^7: Use quaternionic structure
    # Pair as quaternions: R^16 -> H^4
    q = z_sphere.reshape(-1, 4, 4)  # (N, 4, 4) - 4 quaternions

    # First Hopf map: multiply quaternions
    # q1 * conj(q2) gives S^7 points
    q1 = q[:, 0, :] + 1j * q[:, 1, :]  # First complex pair
    q[:, 2, :] + 1j * q[:, 3, :]  # Second complex pair

    # Hopf-like projection to S^3 then to R^3
    w1 = q1[:, 0] + 1j * q1[:, 1]
    w2 = q1[:, 2] + 1j * q1[:, 3]

    # Stereographic projection from S^3 to R^3
    denom = 1 - np.real(w2)
    denom = np.where(np.abs(denom) < 1e-8, 1e-8, denom)

    x = np.real(w1) / denom
    y = np.imag(w1) / denom
    z_out = np.imag(w2) / denom

    result = np.stack([x, y, z_out], axis=1)

    # Clip outliers from stereographic projection
    result = np.clip(result, -10, 10)
    max_val = np.percentile(np.abs(result), 98)
    if max_val > 0:
        result = result / max_val

    return result


def complex_algebraic_projection(z, degree=3):
    """Project using complex algebraic variety structure.

    Treats the embedding as points on an algebraic variety
    and projects to 3D using polynomial relationships.
    """
    z.shape[0]

    # Create complex coordinates
    z_c = z[:, ::2] + 1j * z[:, 1::2]  # (N, 8) complex

    # Use symmetric polynomials for projection
    # These are invariant under permutations, capturing algebraic structure

    # Elementary symmetric polynomials (approximated for efficiency)
    e1 = np.sum(z_c, axis=1)  # Sum
    e2 = np.sum(z_c[:, :-1] * z_c[:, 1:], axis=1)  # Sum of products of pairs
    e3 = np.sum(z_c[:, :-2] * z_c[:, 1:-1] * z_c[:, 2:], axis=1)  # Triples

    # Power sums for additional structure
    p2 = np.sum(z_c**2, axis=1)
    np.sum(z_c**3, axis=1)

    # Combine into 3D coordinates
    x = np.real(e1) + 0.3 * np.real(p2)
    y = np.imag(e1) + 0.3 * np.imag(p2)
    z_out = np.real(e2) + 0.2 * np.real(e3)

    result = np.stack([x, y, z_out], axis=1)
    max_val = np.percentile(np.abs(result), 99)
    if max_val > 0:
        result = result / max_val

    return result


def mirror_symmetry_projection(z):
    """Project using mirror symmetry pairing.

    In string theory, Calabi-Yau manifolds come in mirror pairs.
    This projection emphasizes the mirror structure.
    """
    z.shape[0]

    # Split into two "mirror" halves
    z1 = z[:, :8]
    z2 = z[:, 8:]

    # Create "mirror" coordinates
    # One half uses Kähler structure, other uses complex structure

    # Kähler-like (symplectic)
    k_x = np.sum(z1[:, ::2] * z1[:, 1::2], axis=1)  # Symplectic pairing
    k_y = np.sum(z1[:, ::2] ** 2 - z1[:, 1::2] ** 2, axis=1)  # Quadratic form

    # Complex-like
    c_z = np.sum(z2[:, ::2] * np.cos(z2[:, 1::2]), axis=1)  # Holomorphic-ish

    # Combine with mirror exchange
    x = k_x + 0.5 * np.sum(z2[:, ::2], axis=1)
    y = k_y + 0.5 * np.sum(z2[:, 1::2], axis=1)
    z_out = c_z + 0.3 * np.sum(z1, axis=1)

    result = np.stack([x, y, z_out], axis=1)
    max_val = np.percentile(np.abs(result), 99)
    if max_val > 0:
        result = result / max_val

    return result


def create_static_visualizations(data, output_path):
    """Create static matplotlib visualizations of all projections."""
    z_A = data["z_A"]
    z_B = data["z_B"]
    acc_A = data["acc_A"]
    acc_B = data["acc_B"]

    projections = [
        ("quintic", calabi_yau_quintic_projection, "Quintic Calabi-Yau"),
        ("fermat", calabi_yau_fermat_projection, "Fermat Surface"),
        ("hopf", hopf_fibration_projection, "Hopf Fibration"),
        ("algebraic", complex_algebraic_projection, "Complex Algebraic"),
        ("mirror", mirror_symmetry_projection, "Mirror Symmetry"),
    ]

    # Create comprehensive figure
    fig = plt.figure(figsize=(24, 20))

    for idx, (_name, proj_func, title) in enumerate(projections):
        # VAE-A projection
        proj_A = proj_func(z_A)
        ax = fig.add_subplot(4, 5, idx + 1, projection="3d")
        ax.scatter(
            proj_A[:, 0],
            proj_A[:, 1],
            proj_A[:, 2],
            c=acc_A,
            cmap="viridis",
            s=1,
            alpha=0.6,
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"VAE-A: {title}")
        ax.view_init(elev=20, azim=45)

        # VAE-B projection
        proj_B = proj_func(z_B)
        ax = fig.add_subplot(4, 5, idx + 6, projection="3d")
        ax.scatter(
            proj_B[:, 0],
            proj_B[:, 1],
            proj_B[:, 2],
            c=acc_B,
            cmap="plasma",
            s=1,
            alpha=0.6,
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"VAE-B: {title}")
        ax.view_init(elev=20, azim=45)

    # Add operation-colored versions for Quintic
    proj_A_q = calabi_yau_quintic_projection(z_A)
    proj_B_q = calabi_yau_quintic_projection(z_B)

    # Color by operation index (3-adic structure)
    op_indices = np.arange(len(z_A))

    ax = fig.add_subplot(4, 5, 11, projection="3d")
    ax.scatter(
        proj_A_q[:, 0],
        proj_A_q[:, 1],
        proj_A_q[:, 2],
        c=op_indices,
        cmap="twilight",
        s=1,
        alpha=0.6,
    )
    ax.set_title("VAE-A Quintic (3-adic index)")
    ax.view_init(elev=30, azim=60)

    ax = fig.add_subplot(4, 5, 12, projection="3d")
    ax.scatter(
        proj_B_q[:, 0],
        proj_B_q[:, 1],
        proj_B_q[:, 2],
        c=op_indices,
        cmap="twilight",
        s=1,
        alpha=0.6,
    )
    ax.set_title("VAE-B Quintic (3-adic index)")
    ax.view_init(elev=30, azim=60)

    # Color by first digit (coarse structure)
    first_digit = op_indices % 3
    ax = fig.add_subplot(4, 5, 13, projection="3d")
    ax.scatter(
        proj_A_q[:, 0],
        proj_A_q[:, 1],
        proj_A_q[:, 2],
        c=first_digit,
        cmap="Set1",
        s=2,
        alpha=0.7,
    )
    ax.set_title("VAE-A (digit 0)")
    ax.view_init(elev=30, azim=60)

    # Color by mod 27 (first 3 digits)
    mod27 = op_indices % 27
    ax = fig.add_subplot(4, 5, 14, projection="3d")
    ax.scatter(
        proj_A_q[:, 0],
        proj_A_q[:, 1],
        proj_A_q[:, 2],
        c=mod27,
        cmap="tab20",
        s=2,
        alpha=0.7,
    )
    ax.set_title("VAE-A (mod 27 classes)")
    ax.view_init(elev=30, azim=60)

    # Different view angles for Quintic
    ax = fig.add_subplot(4, 5, 15, projection="3d")
    ax.scatter(
        proj_A_q[:, 0],
        proj_A_q[:, 1],
        proj_A_q[:, 2],
        c=acc_A,
        cmap="RdYlGn",
        s=1,
        alpha=0.6,
    )
    ax.set_title("VAE-A Quintic (top view)")
    ax.view_init(elev=90, azim=0)

    # Phase variations
    for phase_idx, phase in enumerate([0, np.pi / 5, 2 * np.pi / 5, 3 * np.pi / 5]):
        proj_phase = calabi_yau_quintic_projection(z_A, phase=phase)
        ax = fig.add_subplot(4, 5, 16 + phase_idx, projection="3d")
        ax.scatter(
            proj_phase[:, 0],
            proj_phase[:, 1],
            proj_phase[:, 2],
            c=acc_A,
            cmap="viridis",
            s=1,
            alpha=0.6,
        )
        ax.set_title(f"Phase = {phase:.2f}")
        ax.view_init(elev=25, azim=45 + phase_idx * 30)

    plt.tight_layout()
    plt.savefig(
        output_path / "calabi_yau_projections.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved: {output_path / 'calabi_yau_projections.png'}")


def create_high_res_visualization(data, output_path):
    """Create high-resolution single-projection visualizations."""
    z_A = data["z_A"]
    data["z_B"]
    acc_A = data["acc_A"]

    # Quintic projection - multiple angles
    proj = calabi_yau_quintic_projection(z_A)

    fig = plt.figure(figsize=(20, 15))

    angles = [
        (20, 45),
        (20, 135),
        (20, 225),
        (20, 315),
        (60, 45),
        (60, 135),
        (-20, 45),
        (90, 0),
    ]

    for idx, (elev, azim) in enumerate(angles):
        ax = fig.add_subplot(2, 4, idx + 1, projection="3d")
        ax.scatter(
            proj[:, 0],
            proj[:, 1],
            proj[:, 2],
            c=acc_A,
            cmap="viridis",
            s=2,
            alpha=0.7,
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Elev={elev}°, Azim={azim}°")
        ax.view_init(elev=elev, azim=azim)

    plt.suptitle(
        "Calabi-Yau Quintic Projection - VAE-A Latent Space\n(19,683 ternary operations)",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(
        output_path / "calabi_yau_quintic_highres.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved: {output_path / 'calabi_yau_quintic_highres.png'}")


def export_to_csv(data, output_path):
    """Export all projection data to CSV for Three.js."""
    z_A = data["z_A"]
    z_B = data["z_B"]
    acc_A = data["acc_A"]
    acc_B = data["acc_B"]
    data["operations"]

    projections = {
        "quintic": calabi_yau_quintic_projection,
        "fermat": calabi_yau_fermat_projection,
        "hopf": hopf_fibration_projection,
        "algebraic": complex_algebraic_projection,
        "mirror": mirror_symmetry_projection,
    }

    # Create comprehensive dataframe for VAE-A
    df_A = pd.DataFrame()
    df_A["idx"] = np.arange(len(z_A))
    df_A["accuracy"] = acc_A

    # Add original 16D coordinates
    for i in range(16):
        df_A[f"z{i}"] = z_A[:, i]

    # Add all projections
    for name, proj_func in projections.items():
        proj = proj_func(z_A)
        df_A[f"{name}_x"] = proj[:, 0]
        df_A[f"{name}_y"] = proj[:, 1]
        df_A[f"{name}_z"] = proj[:, 2]

    # Add operation encoding for coloring
    df_A["digit_0"] = df_A["idx"] % 3
    df_A["digit_1"] = (df_A["idx"] // 3) % 3
    df_A["digit_2"] = (df_A["idx"] // 9) % 3
    df_A["mod_27"] = df_A["idx"] % 27
    df_A["mod_81"] = df_A["idx"] % 81

    # Save
    csv_path = output_path / "calabi_yau_vae_a.csv"
    df_A.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path} ({len(df_A)} points, {len(df_A.columns)} columns)")

    # Same for VAE-B
    df_B = pd.DataFrame()
    df_B["idx"] = np.arange(len(z_B))
    df_B["accuracy"] = acc_B

    for i in range(16):
        df_B[f"z{i}"] = z_B[:, i]

    for name, proj_func in projections.items():
        proj = proj_func(z_B)
        df_B[f"{name}_x"] = proj[:, 0]
        df_B[f"{name}_y"] = proj[:, 1]
        df_B[f"{name}_z"] = proj[:, 2]

    df_B["digit_0"] = df_B["idx"] % 3
    df_B["digit_1"] = (df_B["idx"] // 3) % 3
    df_B["digit_2"] = (df_B["idx"] // 9) % 3
    df_B["mod_27"] = df_B["idx"] % 27
    df_B["mod_81"] = df_B["idx"] % 81

    csv_path = output_path / "calabi_yau_vae_b.csv"
    df_B.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path} ({len(df_B)} points, {len(df_B.columns)} columns)")

    # Create a smaller sample for initial loading (faster)
    sample_idx = np.random.choice(len(z_A), size=5000, replace=False)
    df_A_sample = df_A.iloc[sample_idx].reset_index(drop=True)
    df_A_sample.to_csv(output_path / "calabi_yau_vae_a_sample.csv", index=False)
    print(f"Saved: {output_path / 'calabi_yau_vae_a_sample.csv'} (5000 points)")

    # Export metadata
    metadata = {
        "total_points": len(z_A),
        "latent_dim": 16,
        "projections": list(projections.keys()),
        "columns": list(df_A.columns),
        "epoch": data["epoch"],
    }
    with open(output_path / "calabi_yau_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: {output_path / 'calabi_yau_metadata.json'}")

    return df_A, df_B


def main():
    output_path = VIZ_DIR / "calabi_yau"
    output_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load best checkpoint (v5.5 has 99.75% coverage)
    # Note: This script expects checkpoint at default location or passed via import
    print("\nLoading v5.5 embeddings...")
    from src.config.paths import CHECKPOINTS_DIR

    data = load_embeddings(str(CHECKPOINTS_DIR / "v5_5" / "latest.pt"), device)
    print(f"Loaded {len(data['z_A'])} embeddings, epoch {data['epoch']}")
    print(f"VAE-A mean accuracy: {data['acc_A'].mean() * 100:.2f}%")
    print(f"VAE-B mean accuracy: {data['acc_B'].mean() * 100:.2f}%")

    # Generate static visualizations
    print("\nGenerating static Calabi-Yau visualizations...")
    create_static_visualizations(data, output_path)
    create_high_res_visualization(data, output_path)

    # Export to CSV
    print("\nExporting to CSV...")
    export_to_csv(data, output_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
