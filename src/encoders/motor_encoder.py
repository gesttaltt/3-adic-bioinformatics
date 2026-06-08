# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Motor-Inspired Encoders with Ternary State Representation.

This module implements encoders inspired by molecular motors like ATP Synthase,
which naturally operate on ternary (3-state) cycles.

Key Features:
- TernaryMotorEncoder: Encodes rotary motor states (α3β3 subunits)
- CyclicStateEmbedding: Embeds discrete cyclic states
- Rotary position encoding for phase-aware representations

Background:
ATP Synthase is a rotary molecular motor with three catalytic sites (β subunits)
that cycle through three conformational states:
- O (Open): Releases ATP
- L (Loose): Binds ADP and Pi
- T (Tight): Catalyzes ATP formation

This 3-state cycle maps naturally to ternary (base-3) number systems.

Usage:
    from src.encoders.motor_encoder import TernaryMotorEncoder

    encoder = TernaryMotorEncoder(embedding_dim=32, n_states=3)
    z = encoder(state_indices, phase_angles)

References:
    - research/clockwork_integration_ideas.md (Idea #7)
    - Boyer (1997) ATP Synthase rotational catalysis mechanism
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.geometry.poincare import project_to_poincare


class RotaryPositionEncoder(nn.Module):
    """Rotary position encoding for cyclic/periodic states.

    Uses sinusoidal encoding on a circle, preserving the cyclic
    nature of rotary motor states.
    """

    def __init__(self, embedding_dim: int, max_states: int = 3):
        """Initialize rotary position encoder.

        Args:
            embedding_dim: Dimension of position embeddings
            max_states: Maximum number of discrete states (default 3 for ternary)
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_states = max_states

        # Precompute frequency bands
        frequencies = torch.exp(
            torch.arange(0, embedding_dim, 2, dtype=torch.float) * -(math.log(10000.0) / embedding_dim)
        )
        self.register_buffer("frequencies", frequencies)

    def forward(self, phase: torch.Tensor) -> torch.Tensor:
        """Encode phase angles into embeddings.

        Args:
            phase: Phase angles in radians, shape (Batch,) or (Batch, SeqLen)

        Returns:
            Rotary embeddings, shape (Batch, Dim) or (Batch, SeqLen, Dim)
        """
        # Ensure phase is at least 2D
        if phase.dim() == 1:
            phase = phase.unsqueeze(-1)
            squeeze_output = True
        else:
            squeeze_output = False

        # Compute sinusoidal embeddings
        # phase: (Batch, SeqLen, 1), frequencies: (Dim//2,)
        angles = phase.unsqueeze(-1) * self.frequencies.unsqueeze(0).unsqueeze(0)

        # Interleave sin and cos
        pe = torch.zeros(*phase.shape[:-1], phase.shape[-1], self.embedding_dim, device=phase.device)
        pe[..., 0::2] = torch.sin(angles)
        pe[..., 1::2] = torch.cos(angles)

        # Squeeze if input was 1D
        if squeeze_output:
            pe = pe.squeeze(-2)

        return pe

    def state_to_phase(self, state_indices: torch.Tensor) -> torch.Tensor:
        """Convert discrete state indices to phase angles.

        Args:
            state_indices: Integer state indices, shape (Batch,)

        Returns:
            Phase angles in [0, 2π), shape (Batch,)
        """
        return 2 * math.pi * state_indices.float() / self.max_states


class TernaryMotorEncoder(nn.Module):
    """Encoder for molecular motor states using ternary representation.

    Models the conformational states of rotary motors like ATP Synthase.
    Each subunit can be in one of three states (O, L, T), and the
    encoder captures both the discrete state and continuous phase.

    The encoder outputs hyperbolic embeddings to capture the hierarchical
    nature of motor assemblies (complex -> subunit -> state).
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        n_subunits: int = 3,
        n_states: int = 3,
        curvature: float = 1.0,
        max_norm: float = 0.95,
    ):
        """Initialize ternary motor encoder.

        Args:
            embedding_dim: Dimension of output embeddings
            n_subunits: Number of motor subunits (3 for ATP Synthase β subunits)
            n_states: Number of states per subunit (3 for O/L/T)
            curvature: Hyperbolic curvature parameter
            max_norm: Maximum norm for Poincaré ball projection
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_subunits = n_subunits
        self.n_states = n_states
        self.curvature = curvature
        self.max_norm = max_norm

        # State embeddings (learnable)
        self.state_embedding = nn.Embedding(n_states, embedding_dim // 2)

        # Subunit embeddings
        self.subunit_embedding = nn.Embedding(n_subunits, embedding_dim // 2)

        # Rotary position encoder for phase
        self.rotary_encoder = RotaryPositionEncoder(embedding_dim, max_states=n_states)

        # Projection network
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        # Initialize with ternary-aware patterns
        self._init_ternary_weights()

    def _init_ternary_weights(self):
        """Initialize embeddings to respect ternary structure.

        States are initialized at 120° intervals on a unit circle,
        reflecting the 3-fold symmetry of ATP Synthase.
        """
        with torch.no_grad():
            # State embeddings at 120° intervals
            for i in range(self.n_states):
                angle = 2 * math.pi * i / self.n_states
                # First half: circular coordinates
                half_dim = self.embedding_dim // 4
                for d in range(half_dim):
                    freq = (d + 1) / half_dim
                    self.state_embedding.weight[i, 2 * d] = math.cos(angle * freq)
                    self.state_embedding.weight[i, 2 * d + 1] = math.sin(angle * freq)

            # Subunit embeddings also at 120° but different phase
            for i in range(self.n_subunits):
                angle = 2 * math.pi * i / self.n_subunits + math.pi / 6  # 30° offset
                half_dim = self.embedding_dim // 4
                for d in range(half_dim):
                    freq = (d + 1) / half_dim
                    self.subunit_embedding.weight[i, 2 * d] = math.cos(angle * freq)
                    self.subunit_embedding.weight[i, 2 * d + 1] = math.sin(angle * freq)

    def forward(
        self,
        state_indices: torch.Tensor,
        subunit_indices: torch.Tensor | None = None,
        phase: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode motor states.

        Args:
            state_indices: State indices per subunit, shape (Batch, n_subunits)
                          or (Batch,) for single subunit
            subunit_indices: Optional subunit indices, shape (Batch, n_subunits)
            phase: Optional continuous phase angles, shape (Batch,)

        Returns:
            Hyperbolic embeddings, shape (Batch, embedding_dim)
        """
        # Handle single subunit case
        if state_indices.dim() == 1:
            state_indices = state_indices.unsqueeze(-1)
        else:
            pass

        batch_size, seq_len = state_indices.shape

        # Get state embeddings
        state_emb = self.state_embedding(state_indices)  # (Batch, SeqLen, Dim//2)

        # Get subunit embeddings
        if subunit_indices is None:
            subunit_indices = torch.arange(seq_len, device=state_indices.device).expand(batch_size, -1)
        subunit_emb = self.subunit_embedding(subunit_indices % self.n_subunits)  # (Batch, SeqLen, Dim//2)

        # Combine state and subunit embeddings
        combined = torch.cat([state_emb, subunit_emb], dim=-1)  # (Batch, SeqLen, Dim)

        # Add rotary position encoding if phase provided
        if phase is not None:
            rotary_emb = self.rotary_encoder(phase)  # (Batch, Dim)
            combined = combined + rotary_emb.unsqueeze(1)

        # Pool across subunits
        pooled = combined.mean(dim=1)  # (Batch, Dim)

        # Add phase-aware features
        if phase is not None:
            phase_emb = self.rotary_encoder(phase)
            pooled = torch.cat([pooled, phase_emb], dim=-1)  # (Batch, Dim*2)
        else:
            # Use mean phase from states
            mean_phase = self.rotary_encoder.state_to_phase(state_indices.float().mean(dim=-1))
            phase_emb = self.rotary_encoder(mean_phase)
            pooled = torch.cat([pooled, phase_emb], dim=-1)

        # Project to final dimension
        output = self.projector(pooled)

        # Project to Poincaré ball
        output = project_to_poincare(output, c=self.curvature, max_norm=self.max_norm)

        return output

    def get_state_transitions(self) -> torch.Tensor:
        """Get embedding distances between consecutive states.

        Returns:
            Transition distances, shape (n_states,)
        """
        with torch.no_grad():
            distances = []
            for i in range(self.n_states):
                j = (i + 1) % self.n_states
                dist = torch.norm(self.state_embedding.weight[i] - self.state_embedding.weight[j])
                distances.append(dist)
            return torch.stack(distances)

    def encode_rotation_cycle(
        self,
        n_steps: int = 12,
    ) -> torch.Tensor:
        """Encode a full rotation cycle for visualization.

        Args:
            n_steps: Number of steps in the cycle

        Returns:
            Embeddings for each step, shape (n_steps, embedding_dim)
        """
        phases = torch.linspace(0, 2 * math.pi, n_steps)

        embeddings = []
        for phase in phases:
            # Determine state indices based on phase
            state_idx = int((phase / (2 * math.pi)) * self.n_states) % self.n_states
            states = torch.tensor([[state_idx] * self.n_subunits])
            emb = self.forward(states, phase=phase.unsqueeze(0))
            embeddings.append(emb)

        return torch.cat(embeddings, dim=0)


class ATPSynthaseEncoder(TernaryMotorEncoder):
    """Specialized encoder for ATP Synthase molecular motor.

    Extends TernaryMotorEncoder with ATP Synthase-specific features:
    - α3β3 hexamer structure
    - γ-subunit central stalk rotation
    - Proton-driven rotation coupling
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        curvature: float = 1.0,
    ):
        """Initialize ATP Synthase encoder.

        Args:
            embedding_dim: Dimension of output embeddings
            curvature: Hyperbolic curvature parameter
        """
        # ATP Synthase has 3 β subunits (catalytic) and 3 α subunits (regulatory)
        super().__init__(
            embedding_dim=embedding_dim,
            n_subunits=3,  # 3 β subunits
            n_states=3,  # O, L, T states
            curvature=curvature,
        )

        # Additional embedding for proton count (0-3 protons per rotation step)
        self.proton_embedding = nn.Embedding(4, embedding_dim // 4)

        # γ-subunit rotation angle encoding
        self.gamma_encoder = RotaryPositionEncoder(embedding_dim // 4, max_states=360)

    def forward(
        self,
        beta_states: torch.Tensor,
        gamma_angle: torch.Tensor | None = None,
        proton_count: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode ATP Synthase state.

        Args:
            beta_states: States of β subunits, shape (Batch, 3)
                        Values: 0=Open, 1=Loose, 2=Tight
            gamma_angle: Rotation angle of γ subunit in degrees, shape (Batch,)
            proton_count: Number of protons translocated, shape (Batch,)

        Returns:
            Hyperbolic embeddings, shape (Batch, embedding_dim)
        """
        # Convert gamma angle to phase if provided
        if gamma_angle is not None:
            phase = gamma_angle * math.pi / 180  # Convert to radians
        else:
            phase = None

        # Get base motor encoding
        base_encoding = super().forward(beta_states, phase=phase)

        return base_encoding


__all__ = [
    "TernaryMotorEncoder",
    "ATPSynthaseEncoder",
    "RotaryPositionEncoder",
]
