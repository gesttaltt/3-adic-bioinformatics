# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Shared mathematical projection functions for Calabi-Yau visualizations.

Extracted from calabi_yau_v58_fast.py and calabi_yau_v58_extended.py
to eliminate code duplication (D4 from DUPLICATION_REPORT).

These projections map high-dimensional latent embeddings to 3D for visualization.
"""

import numpy as np
import torch


def _to_numpy(z: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert input to numpy array."""
    return z.numpy() if torch.is_tensor(z) else z


def _normalize(z: np.ndarray) -> np.ndarray:
    """L2 normalize each row."""
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)


def _to_complex_coords(z_norm: np.ndarray, n_pairs: int) -> list:
    """Convert normalized real coordinates to complex pairs."""
    return [z_norm[:, 2 * i] + 1j * z_norm[:, 2 * i + 1] for i in range(n_pairs)]


def quintic_fibration(z: np.ndarray | torch.Tensor) -> np.ndarray:
    """Calabi-Yau quintic threefold: z1^5 + z2^5 + z3^5 + z4^5 + z5^5 = 0

    Projects high-dimensional embeddings onto a 3D representation of the
    quintic Calabi-Yau manifold fibration structure.

    Args:
        z: Embeddings of shape (N, dim) where dim >= 4

    Returns:
        3D coordinates of shape (N, 3)
    """
    z = _to_numpy(z)
    z_norm = _normalize(z)
    dim = z.shape[1]
    n_complex = min(5, dim // 2)

    w = _to_complex_coords(z_norm, n_complex)
    constraint = sum(c**5 for c in w)

    x = np.real(w[0] * np.conj(w[1]))
    y = np.imag(w[0] * np.conj(w[1]))

    phase = np.angle(w[0]) + np.angle(w[2]) if len(w) > 2 else np.angle(w[0])
    z_coord = np.sin(phase * 2.5) * np.abs(constraint) * 0.3

    if len(w) >= 4:
        z_coord += 0.2 * np.real(w[2] * w[3])

    return np.column_stack([x, y, z_coord])


def hopf_fibration(z: np.ndarray | torch.Tensor) -> np.ndarray:
    """Hopf fibration S^(2n-1) -> CP^(n-1)

    Projects embeddings using the Hopf fibration structure, revealing
    the fiber bundle topology.

    Args:
        z: Embeddings of shape (N, dim) where dim >= 4

    Returns:
        3D coordinates of shape (N, 3)
    """
    z = _to_numpy(z)
    z_norm = _normalize(z)
    dim = z.shape[1]
    n_pairs = min(8, dim // 2)

    w = _to_complex_coords(z_norm, n_pairs)

    # Base space (S^2)
    z1, z2 = w[0], w[1]
    x = 2 * np.real(z1 * np.conj(z2))
    y = 2 * np.imag(z1 * np.conj(z2))
    z_base = np.abs(z1) ** 2 - np.abs(z2) ** 2

    # Fiber contribution
    if len(w) >= 4:
        fiber_phase = np.angle(w[2]) - np.angle(w[3])
        z_coord = z_base + 0.15 * np.sin(3 * fiber_phase)
    else:
        z_coord = z_base

    return np.column_stack([x, y, z_coord])


def k3_surface(z: np.ndarray | torch.Tensor) -> np.ndarray:
    """K3 surface projection.

    Projects onto a representation related to K3 surfaces, which are
    important in string compactification.

    Args:
        z: Embeddings of shape (N, dim) where dim >= 8

    Returns:
        3D coordinates of shape (N, 3)
    """
    z = _to_numpy(z)
    z_norm = _normalize(z)
    dim = z.shape[1]
    n_pairs = min(4, dim // 2)

    w = _to_complex_coords(z_norm, n_pairs)

    # Quartic in CP^3
    quartic = sum(c**4 for c in w)

    x = np.real(w[0] * w[1])
    y = np.imag(w[0] * w[1])
    z_coord = np.real(quartic) * 0.5

    if len(w) >= 4:
        z_coord += 0.2 * np.imag(w[2] * w[3])

    return np.column_stack([x, y, z_coord])


def mirror_symmetry(z: np.ndarray | torch.Tensor) -> np.ndarray:
    """Mirror symmetry projection.

    Projects using mirror symmetry relations between complex and
    Kahler moduli spaces.

    Args:
        z: Embeddings of shape (N, dim) where dim >= 8

    Returns:
        3D coordinates of shape (N, 3)
    """
    z = _to_numpy(z)
    z_norm = _normalize(z)
    dim = z.shape[1]
    n_pairs = min(5, dim // 2)

    w = _to_complex_coords(z_norm, n_pairs)

    # Complex structure moduli
    tau = w[0] / (w[1] + 1e-8)

    x = np.real(tau)
    y = np.imag(tau)

    # Kahler moduli contribution
    if len(w) >= 4:
        rho = w[2] / (w[3] + 1e-8)
        z_coord = np.abs(rho) * np.sin(np.angle(rho) + np.angle(tau))
    else:
        z_coord = np.abs(tau) * 0.3

    return np.column_stack([x, y, z_coord])


def fermat_surface(z: np.ndarray | torch.Tensor, degree: int = 4) -> np.ndarray:
    """Fermat surface z1^d + z2^d + z3^d + z4^d = 0

    Projects onto the Fermat hypersurface of specified degree.

    Args:
        z: Embeddings of shape (N, dim) where dim >= 8
        degree: Degree of the Fermat equation (default 4)

    Returns:
        3D coordinates of shape (N, 3)
    """
    z = _to_numpy(z)
    z_norm = _normalize(z)
    dim = z.shape[1]
    n_pairs = min(4, dim // 2)

    w = _to_complex_coords(z_norm, n_pairs)

    constraint = sum(c**degree for c in w)

    x = np.real(w[0] * np.conj(w[1]))
    y = np.imag(w[0] * np.conj(w[1]))
    z_coord = np.abs(constraint) * np.cos(np.angle(constraint))

    return np.column_stack([x, y, z_coord])


def torus_fibration(z: np.ndarray | torch.Tensor) -> np.ndarray:
    """Elliptic fibration (torus fibers over base).

    Projects using the structure of an elliptic fibration, where
    torus fibers vary over a base space.

    Args:
        z: Embeddings of shape (N, dim) where dim >= 6

    Returns:
        3D coordinates of shape (N, 3)
    """
    z = _to_numpy(z)
    z_norm = _normalize(z)
    dim = z.shape[1]
    n_pairs = min(4, dim // 2)

    w = _to_complex_coords(z_norm, n_pairs)

    # Base coordinate
    base_r = np.abs(w[0])
    base_theta = np.angle(w[0])

    # Fiber coordinates (torus)
    fiber_theta = np.angle(w[1]) if len(w) > 1 else 0
    fiber_phi = np.angle(w[2]) if len(w) > 2 else 0

    # Torus embedding with varying major radius
    R = 1.0 + 0.3 * base_r
    r = 0.3

    x = (R + r * np.cos(fiber_phi)) * np.cos(base_theta + fiber_theta)
    y = (R + r * np.cos(fiber_phi)) * np.sin(base_theta + fiber_theta)
    z_coord = r * np.sin(fiber_phi)

    return np.column_stack([x, y, z_coord])


# Dictionary for easy access to all projections
PROJECTIONS = {
    "quintic": quintic_fibration,
    "hopf": hopf_fibration,
    "k3": k3_surface,
    "mirror": mirror_symmetry,
    "fermat": fermat_surface,
    "torus": torus_fibration,
}
