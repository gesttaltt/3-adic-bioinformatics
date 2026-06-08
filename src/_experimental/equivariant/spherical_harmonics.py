# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Spherical harmonics utilities for equivariant networks.

This module provides implementations of spherical harmonics for use in
SO(3)-equivariant neural networks. Supports both native PyTorch implementation
and optional e3nn library for optimized operations.

References:
    - Weiler et al., "3D Steerable CNNs" (2018)
    - Thomas et al., "Tensor field networks" (2018)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

# Try to import e3nn for optimized spherical harmonics
try:
    import e3nn
    from e3nn import o3

    HAS_E3NN = True
except ImportError:
    HAS_E3NN = False


def factorial(n: int) -> int:
    """Compute factorial of n."""
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def double_factorial(n: int) -> int:
    """Compute double factorial n!! = n * (n-2) * (n-4) * ..."""
    if n <= 0:
        return 1
    result = 1
    while n > 0:
        result *= n
        n -= 2
    return result


def associated_legendre(l: int, m: int, x: Tensor) -> Tensor:
    """Compute associated Legendre polynomial P_l^m(x).

    Args:
        l: Degree (non-negative integer)
        m: Order (-l <= m <= l)
        x: Input values in range [-1, 1]

    Returns:
        P_l^m(x) values
    """
    # Use absolute value of m for computation
    abs_m = abs(m)

    if abs_m > l:
        return torch.zeros_like(x)

    # Start with P_m^m using recurrence
    pmm = torch.ones_like(x)
    if abs_m > 0:
        somx2 = torch.sqrt((1 - x) * (1 + x))
        fact = 1.0
        for _i in range(1, abs_m + 1):
            pmm = -pmm * fact * somx2
            fact += 2.0

    if l == abs_m:
        result = pmm
    else:
        # Compute P_{m+1}^m
        pmmp1 = x * (2 * abs_m + 1) * pmm
        if l == abs_m + 1:
            result = pmmp1
        else:
            # Use upward recurrence for higher l
            for ll in range(abs_m + 2, l + 1):
                pll = (x * (2 * ll - 1) * pmmp1 - (ll + abs_m - 1) * pmm) / (ll - abs_m)
                pmm = pmmp1
                pmmp1 = pll
            result = pmmp1

    # Apply sign for negative m
    if m < 0:
        sign = (-1) ** abs_m
        factor = factorial(l - abs_m) / factorial(l + abs_m)
        result = sign * factor * result

    return result


def spherical_harmonics_manual(l: int, m: int, theta: Tensor, phi: Tensor) -> Tensor:
    """Compute spherical harmonic Y_l^m(theta, phi).

    Uses real spherical harmonics convention.

    Args:
        l: Degree (non-negative integer)
        m: Order (-l <= m <= l)
        theta: Polar angle (0 to pi)
        phi: Azimuthal angle (0 to 2*pi)

    Returns:
        Y_l^m(theta, phi) values (real-valued)
    """
    # Normalization factor
    norm = math.sqrt((2 * l + 1) / (4 * math.pi))
    norm *= math.sqrt(factorial(l - abs(m)) / factorial(l + abs(m)))

    # Associated Legendre polynomial
    plm = associated_legendre(l, abs(m), torch.cos(theta))

    # Real spherical harmonics
    if m > 0:
        result = norm * math.sqrt(2) * plm * torch.cos(m * phi)
    elif m < 0:
        result = norm * math.sqrt(2) * plm * torch.sin(abs(m) * phi)
    else:
        result = norm * plm

    return result


class SphericalHarmonics(nn.Module):
    """Spherical harmonics computation module.

    Computes spherical harmonics up to degree lmax. Uses e3nn library
    if available, otherwise falls back to manual implementation.

    Args:
        lmax: Maximum degree of spherical harmonics
        normalize: Whether to use normalized spherical harmonics
        use_e3nn: Whether to use e3nn library (if available)
    """

    def __init__(
        self,
        lmax: int = 2,
        normalize: bool = True,
        use_e3nn: bool = True,
    ):
        super().__init__()
        self.lmax = lmax
        self.normalize = normalize
        self.use_e3nn = use_e3nn and HAS_E3NN

        # Number of spherical harmonics up to lmax
        self.n_harmonics = (lmax + 1) ** 2

        if self.use_e3nn:
            # e3nn irreps string for spherical harmonics
            self.irreps = o3.Irreps.spherical_harmonics(lmax)

    @property
    def output_dim(self) -> int:
        """Number of output spherical harmonic coefficients."""
        return self.n_harmonics

    def forward(self, vectors: Tensor) -> Tensor:
        """Compute spherical harmonics for direction vectors.

        Args:
            vectors: Direction vectors of shape (..., 3)

        Returns:
            Spherical harmonic values of shape (..., n_harmonics)
        """
        if self.use_e3nn:
            return self._forward_e3nn(vectors)
        else:
            return self._forward_manual(vectors)

    def _forward_e3nn(self, vectors: Tensor) -> Tensor:
        """Compute using e3nn library."""
        # e3nn expects normalized vectors
        norms = torch.linalg.norm(vectors, dim=-1, keepdim=True)
        normalized = vectors / (norms + 1e-8)

        # Compute spherical harmonics
        sh = o3.spherical_harmonics(self.irreps, normalized, normalize=self.normalize)
        return sh

    def _forward_manual(self, vectors: Tensor) -> Tensor:
        """Compute using manual implementation."""
        # Convert to spherical coordinates
        x, y, z = vectors[..., 0], vectors[..., 1], vectors[..., 2]
        r = torch.sqrt(x**2 + y**2 + z**2)
        r = torch.clamp(r, min=1e-8)

        theta = torch.acos(torch.clamp(z / r, -1 + 1e-7, 1 - 1e-7))
        phi = torch.atan2(y, x)

        # Compute all spherical harmonics
        results = []
        for l in range(self.lmax + 1):
            for m in range(-l, l + 1):
                ylm = spherical_harmonics_manual(l, m, theta, phi)
                results.append(ylm)

        return torch.stack(results, dim=-1)


class ClebschGordanCoefficients:
    """Clebsch-Gordan coefficients for coupling spherical harmonics.

    These coefficients describe how to combine two spherical harmonics
    into a third, preserving rotational equivariance.
    """

    def __init__(self, lmax: int = 2):
        self.lmax = lmax
        self._cache: dict[tuple[int, int, int, int, int, int], float] = {}

    def __call__(self, l1: int, m1: int, l2: int, m2: int, l3: int, m3: int) -> float:
        """Get Clebsch-Gordan coefficient <l1,m1;l2,m2|l3,m3>.

        Args:
            l1, m1: First angular momentum
            l2, m2: Second angular momentum
            l3, m3: Total angular momentum

        Returns:
            Clebsch-Gordan coefficient value
        """
        key = (l1, m1, l2, m2, l3, m3)
        if key in self._cache:
            return self._cache[key]

        # Selection rules
        if m1 + m2 != m3:
            return 0.0
        if l3 < abs(l1 - l2) or l3 > l1 + l2:
            return 0.0

        # Compute using Racah formula
        value = self._compute_cg(l1, m1, l2, m2, l3, m3)
        self._cache[key] = value
        return value

    def _compute_cg(self, l1: int, m1: int, l2: int, m2: int, l3: int, m3: int) -> float:
        """Compute CG coefficient using Racah formula."""
        # Check validity
        if abs(m1) > l1 or abs(m2) > l2 or abs(m3) > l3:
            return 0.0

        # Prefactor
        prefactor = math.sqrt(
            (2 * l3 + 1)
            * factorial(l1 + l2 - l3)
            * factorial(l1 - l2 + l3)
            * factorial(-l1 + l2 + l3)
            / factorial(l1 + l2 + l3 + 1)
        )
        prefactor *= math.sqrt(
            factorial(l3 + m3)
            * factorial(l3 - m3)
            * factorial(l1 - m1)
            * factorial(l1 + m1)
            * factorial(l2 - m2)
            * factorial(l2 + m2)
        )

        # Sum over k
        kmin = max(0, l2 - l3 - m1, l1 - l3 + m2)
        kmax = min(l1 + l2 - l3, l1 - m1, l2 + m2)

        total = 0.0
        for k in range(kmin, kmax + 1):
            sign = (-1) ** k
            denom = (
                factorial(k)
                * factorial(l1 + l2 - l3 - k)
                * factorial(l1 - m1 - k)
                * factorial(l2 + m2 - k)
                * factorial(l3 - l2 + m1 + k)
                * factorial(l3 - l1 - m2 + k)
            )
            if denom != 0:
                total += sign / denom

        return prefactor * total


def wigner_d_matrix(l: int, alpha: Tensor, beta: Tensor, gamma: Tensor) -> Tensor:
    """Compute Wigner D-matrix for rotation.

    The Wigner D-matrix describes how spherical harmonics transform
    under rotations parameterized by Euler angles.

    Args:
        l: Angular momentum quantum number
        alpha: First Euler angle (rotation about z)
        beta: Second Euler angle (rotation about y')
        gamma: Third Euler angle (rotation about z'')

    Returns:
        D-matrix of shape (..., 2l+1, 2l+1)
    """
    # Small d-matrix (rotation about y-axis)
    d = _small_d_matrix(l, beta)

    # Full D-matrix with phases from z-rotations
    m = torch.arange(-l, l + 1, device=alpha.device, dtype=alpha.dtype)
    mp = torch.arange(-l, l + 1, device=alpha.device, dtype=alpha.dtype)

    # Expand for broadcasting
    alpha_exp = alpha[..., None, None]
    gamma_exp = gamma[..., None, None]
    m_exp = m[None, :]
    mp_exp = mp[:, None]

    phase = torch.exp(-1j * (m_exp * alpha_exp + mp_exp * gamma_exp))
    D = phase * d

    # Return real part for real spherical harmonics
    return D.real


def _small_d_matrix(l: int, beta: Tensor) -> Tensor:
    """Compute small Wigner d-matrix for rotation about y-axis."""
    size = 2 * l + 1
    d = torch.zeros(*beta.shape, size, size, device=beta.device, dtype=beta.dtype)

    cos_half = torch.cos(beta / 2)
    sin_half = torch.sin(beta / 2)

    for m in range(-l, l + 1):
        for mp in range(-l, l + 1):
            # Wigner formula
            s_min = max(0, m - mp)
            s_max = min(l + m, l - mp)

            total = torch.zeros_like(beta)
            for s in range(s_min, s_max + 1):
                sign = (-1) ** (mp - m + s)
                binom1 = factorial(l + m) // (factorial(l + m - s) * factorial(s))
                binom2 = factorial(l - m) // (factorial(l - mp - s) * factorial(s - m + mp))

                power_cos = 2 * l + m - mp - 2 * s
                power_sin = mp - m + 2 * s

                term = sign * binom1 * binom2 * cos_half**power_cos * sin_half**power_sin
                total = total + term

            d[..., m + l, mp + l] = total

    return d
