# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Circadian Cycle Encoder with Toroidal Topology.

This module implements encoders for circadian rhythm patterns using
toroidal (S¹ × S¹) geometry to capture the doubly-periodic nature
of biological oscillators.

Key Features:
- ToroidalEmbedding: Maps to a torus (product of two circles)
- CircadianCycleEncoder: Encodes phosphorylation cycles like KaiC
- Phase-locked oscillator patterns

Background:
The KaiC protein acts as a biological clock via phosphorylation cycles.
The phosphorylation state cycles through a 24-hour period, which can be
modeled on a torus where:
- One circle represents the 24-hour day/night cycle
- The other represents the phosphorylation state cycle

Usage:
    from src.encoders.circadian_encoder import CircadianCycleEncoder

    encoder = CircadianCycleEncoder(embedding_dim=32)
    z = encoder(time_of_day, phosphorylation_state)

References:
    - research/clockwork_integration_ideas.md (Idea #13)
    - Nakajima et al. (2005) KaiC phosphorylation cycle
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ToroidalEmbedding(nn.Module):
    """Embedding on a torus (S¹ × S¹).

    Maps two angular coordinates to a flat embedding space while
    preserving the toroidal topology through circular harmonics.

    The torus is parameterized by two angles (θ, φ) ∈ [0, 2π) × [0, 2π).
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        n_harmonics: int = 4,
        major_radius: float = 1.0,
        minor_radius: float = 0.5,
    ):
        """Initialize toroidal embedding.

        Args:
            embedding_dim: Output embedding dimension
            n_harmonics: Number of Fourier harmonics per angle
            major_radius: Radius of the torus centerline (R)
            minor_radius: Radius of the tube (r)
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_harmonics = n_harmonics
        self.major_radius = major_radius
        self.minor_radius = minor_radius

        # Learnable harmonic weights
        self.theta_weights = nn.Parameter(torch.randn(n_harmonics) * 0.1)
        self.phi_weights = nn.Parameter(torch.randn(n_harmonics) * 0.1)

        # Projection to embedding dimension
        harmonic_dim = 4 * n_harmonics + 3  # sin/cos for each harmonic + 3D coords
        self.projector = nn.Linear(harmonic_dim, embedding_dim)

    def _compute_harmonics(self, angle: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Compute weighted Fourier harmonics.

        Args:
            angle: Angles in radians, shape (Batch,)
            weights: Harmonic weights, shape (n_harmonics,)

        Returns:
            Harmonic features, shape (Batch, 2*n_harmonics)
        """
        harmonics = []
        for k in range(1, self.n_harmonics + 1):
            harmonics.append(weights[k - 1] * torch.sin(k * angle))
            harmonics.append(weights[k - 1] * torch.cos(k * angle))
        return torch.stack(harmonics, dim=-1)

    def to_3d_torus(self, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Convert toroidal coordinates to 3D Euclidean coordinates.

        Args:
            theta: Major angle (around the hole), shape (Batch,)
            phi: Minor angle (around the tube), shape (Batch,)

        Returns:
            3D coordinates, shape (Batch, 3)
        """
        R = self.major_radius
        r = self.minor_radius

        x = (R + r * torch.cos(phi)) * torch.cos(theta)
        y = (R + r * torch.cos(phi)) * torch.sin(theta)
        z = r * torch.sin(phi)

        return torch.stack([x, y, z], dim=-1)

    def forward(self, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Embed points on the torus.

        Args:
            theta: Major angle in radians, shape (Batch,)
            phi: Minor angle in radians, shape (Batch,)

        Returns:
            Embeddings, shape (Batch, embedding_dim)
        """
        # Compute 3D torus coordinates
        coords_3d = self.to_3d_torus(theta, phi)

        # Compute harmonic features
        theta_harmonics = self._compute_harmonics(theta, self.theta_weights)
        phi_harmonics = self._compute_harmonics(phi, self.phi_weights)

        # Concatenate all features
        features = torch.cat([coords_3d, theta_harmonics, phi_harmonics], dim=-1)

        # Project to embedding dimension
        return self.projector(features)


class CircadianCycleEncoder(nn.Module):
    """Encoder for circadian rhythm patterns on a toroidal manifold.

    Models biological oscillators like the KaiC phosphorylation cycle
    using a toroidal topology where:
    - θ (theta): Time of day (24-hour cycle)
    - φ (phi): Internal biochemical state (phosphorylation cycle)

    The encoder captures phase relationships between environmental
    time and internal molecular state.
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        n_phospho_states: int = 4,
        period_hours: float = 24.0,
        n_harmonics: int = 4,
    ):
        """Initialize circadian cycle encoder.

        Args:
            embedding_dim: Output embedding dimension
            n_phospho_states: Number of discrete phosphorylation states
            period_hours: Circadian period in hours
            n_harmonics: Number of Fourier harmonics
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_phospho_states = n_phospho_states
        self.period_hours = period_hours

        # Toroidal embedding
        self.torus = ToroidalEmbedding(
            embedding_dim=embedding_dim // 2,
            n_harmonics=n_harmonics,
        )

        # Phosphorylation state embedding
        self.phospho_embedding = nn.Embedding(n_phospho_states, embedding_dim // 4)

        # Phase coupling parameters (learnable)
        self.phase_coupling = nn.Parameter(torch.tensor(0.5))

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(embedding_dim // 2 + embedding_dim // 4, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize phosphorylation embeddings on a circle."""
        with torch.no_grad():
            for i in range(self.n_phospho_states):
                angle = 2 * math.pi * i / self.n_phospho_states
                dim = self.embedding_dim // 4
                for d in range(dim // 2):
                    self.phospho_embedding.weight[i, 2 * d] = math.cos(angle * (d + 1))
                    self.phospho_embedding.weight[i, 2 * d + 1] = math.sin(angle * (d + 1))

    def time_to_phase(self, time_hours: torch.Tensor) -> torch.Tensor:
        """Convert time in hours to phase angle.

        Args:
            time_hours: Time in hours [0, period), shape (Batch,)

        Returns:
            Phase angle in [0, 2π), shape (Batch,)
        """
        return 2 * math.pi * (time_hours % self.period_hours) / self.period_hours

    def phospho_to_phase(self, phospho_state: torch.Tensor) -> torch.Tensor:
        """Convert phosphorylation state to phase angle.

        Args:
            phospho_state: Discrete state index, shape (Batch,)

        Returns:
            Phase angle in [0, 2π), shape (Batch,)
        """
        return 2 * math.pi * phospho_state.float() / self.n_phospho_states

    def forward(
        self,
        time_hours: torch.Tensor,
        phospho_state: torch.Tensor | None = None,
        phospho_continuous: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode circadian state.

        Args:
            time_hours: Time of day in hours, shape (Batch,)
            phospho_state: Discrete phosphorylation state, shape (Batch,)
            phospho_continuous: Continuous phosphorylation level [0, 1], shape (Batch,)

        Returns:
            Embeddings, shape (Batch, embedding_dim)
        """
        # Convert time to phase
        theta = self.time_to_phase(time_hours)

        # Determine phosphorylation phase
        if phospho_continuous is not None:
            phi = 2 * math.pi * phospho_continuous
        elif phospho_state is not None:
            phi = self.phospho_to_phase(phospho_state)
        else:
            # Default: phase-locked to time
            phi = theta * self.phase_coupling

        # Get toroidal embedding
        torus_emb = self.torus(theta, phi)

        # Get phosphorylation embedding if discrete state provided
        if phospho_state is not None:
            phospho_emb = self.phospho_embedding(phospho_state)
        else:
            # Use continuous approximation
            if phospho_continuous is not None:
                # Interpolate between embeddings
                state_float = phospho_continuous * self.n_phospho_states
                state_low = state_float.long() % self.n_phospho_states
                state_high = (state_low + 1) % self.n_phospho_states
                alpha = state_float - state_float.floor()

                emb_low = self.phospho_embedding(state_low)
                emb_high = self.phospho_embedding(state_high)
                phospho_emb = (1 - alpha.unsqueeze(-1)) * emb_low + alpha.unsqueeze(-1) * emb_high
            else:
                # Default embedding
                phospho_emb = self.phospho_embedding(torch.zeros_like(time_hours, dtype=torch.long))

        # Combine embeddings
        combined = torch.cat([torus_emb, phospho_emb], dim=-1)

        # Project to output dimension
        return self.output_proj(combined)

    def get_phase_trajectory(
        self,
        n_timepoints: int = 48,
        phospho_sequence: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate embeddings for a full circadian cycle.

        Args:
            n_timepoints: Number of timepoints to sample
            phospho_sequence: Optional phosphorylation states, shape (n_timepoints,)

        Returns:
            (embeddings, times) where embeddings is (n_timepoints, embedding_dim)
        """
        times = torch.linspace(0, self.period_hours, n_timepoints)

        if phospho_sequence is None:
            # Phase-locked phosphorylation cycle
            phospho_phases = (times / self.period_hours * self.n_phospho_states).long() % self.n_phospho_states
        else:
            phospho_phases = phospho_sequence

        embeddings = self.forward(times, phospho_state=phospho_phases)

        return embeddings, times

    def compute_phase_coherence(
        self,
        time_hours: torch.Tensor,
        phospho_state: torch.Tensor,
    ) -> torch.Tensor:
        """Compute phase coherence between time and phosphorylation.

        Phase coherence measures how well the internal state is synchronized
        with the environmental cycle. Values near 1 indicate strong coupling.

        Args:
            time_hours: Time of day, shape (Batch,)
            phospho_state: Phosphorylation state, shape (Batch,)

        Returns:
            Phase coherence, shape (Batch,)
        """
        theta = self.time_to_phase(time_hours)
        phi = self.phospho_to_phase(phospho_state)

        # Expected phase relationship
        expected_phi = theta * self.phase_coupling

        # Circular distance
        phase_diff = phi - expected_phi
        coherence = torch.cos(phase_diff)

        return coherence


class KaiCClockEncoder(CircadianCycleEncoder):
    """Specialized encoder for the KaiC circadian clock protein.

    KaiC has four phosphorylation sites (S431 and T432 on each of two domains)
    creating a 4-state cycle: U (unphosphorylated) → T → ST → S → U

    The phosphorylation cycle takes ~24 hours in vitro with KaiA and KaiB.
    """

    def __init__(self, embedding_dim: int = 64):
        """Initialize KaiC encoder.

        Args:
            embedding_dim: Output embedding dimension
        """
        super().__init__(
            embedding_dim=embedding_dim,
            n_phospho_states=4,  # U, T, ST, S
            period_hours=24.0,
            n_harmonics=6,
        )

        # State names for reference
        self.state_names = ["U", "T", "ST", "S"]

        # KaiA/KaiB interaction modifiers
        self.kaia_effect = nn.Parameter(torch.tensor(1.0))
        self.kaib_effect = nn.Parameter(torch.tensor(-0.5))

    def forward(
        self,
        time_hours: torch.Tensor,
        phospho_state: torch.Tensor | None = None,
        phospho_continuous: torch.Tensor | None = None,
        kaia_level: torch.Tensor | None = None,
        kaib_level: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode KaiC state with optional KaiA/KaiB modulation.

        Args:
            time_hours: Time of day in hours, shape (Batch,)
            phospho_state: Phosphorylation state (0=U, 1=T, 2=ST, 3=S)
            phospho_continuous: Continuous phosphorylation level [0, 1], shape (Batch,)
            kaia_level: KaiA concentration [0, 1], shape (Batch,)
            kaib_level: KaiB concentration [0, 1], shape (Batch,)

        Returns:
            Embeddings, shape (Batch, embedding_dim)
        """
        # Get base encoding
        base_emb = super().forward(time_hours, phospho_state=phospho_state, phospho_continuous=phospho_continuous)

        # Modulate by KaiA/KaiB if provided
        if kaia_level is not None or kaib_level is not None:
            modulation = torch.ones_like(time_hours)
            if kaia_level is not None:
                modulation = modulation + self.kaia_effect * kaia_level
            if kaib_level is not None:
                modulation = modulation + self.kaib_effect * kaib_level
            base_emb = base_emb * modulation.unsqueeze(-1)

        return base_emb


__all__ = [
    "ToroidalEmbedding",
    "CircadianCycleEncoder",
    "KaiCClockEncoder",
]
