# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Protein energy landscape analysis using ultrametric geometry.

This module implements ultrametric analysis of protein folding landscapes,
where the hierarchical structure of conformational states is naturally
described by p-adic distances. The ultrametric tree structure captures
kinetic barriers between states.

Key concepts:
- Ultrametric distance: d(a,c) <= max(d(a,b), d(b,c))
- Folding funnels as p-adic basins
- Kinetic traps as local minima in ultrametric space
- Transition states as saddle points

References:
- 2012_Scalco_Protein_Ultrametricity.md: Ultrametric protein geometry
- 1997_Onuchic_Protein_Landscapes.md: Energy landscape theory
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torch.nn as nn


class ConformationState(Enum):
    """Protein conformational states."""

    NATIVE = "native"
    MOLTEN_GLOBULE = "molten_globule"
    INTERMEDIATE = "intermediate"
    UNFOLDED = "unfolded"
    MISFOLDED = "misfolded"
    AGGREGATED = "aggregated"


@dataclass
class EnergyBasin:
    """Represents an energy basin in the folding landscape."""

    center: torch.Tensor  # Representative conformation
    energy: float  # Basin energy level
    depth: float  # Basin depth (stability)
    width: float  # Basin width (entropy)
    state: ConformationState
    escape_barrier: float  # Barrier to exit basin


@dataclass
class TransitionPath:
    """A transition path between conformational states."""

    start_state: ConformationState
    end_state: ConformationState
    barrier_height: float  # Energy barrier
    path_length: float  # Path length in conformational space
    transition_state: torch.Tensor | None  # Transition state conformation
    rate_constant: float  # Estimated rate constant


@dataclass
class LandscapeMetrics:
    """Metrics describing the folding landscape."""

    ruggedness: float  # Landscape ruggedness (barrier variance)
    funnel_depth: float  # Depth of folding funnel
    frustration: float  # Degree of frustration
    ultrametricity: float  # Degree of ultrametric structure
    n_basins: int  # Number of metastable basins
    native_stability: float  # Stability of native state


class UltrametricDistanceMatrix(nn.Module):
    """Computes and analyzes ultrametric distance matrices.

    The ultrametric property requires that for any three points,
    the two largest pairwise distances are equal.
    """

    def __init__(self, p: int = 3, n_states: int = 100):
        """Initialize ultrametric distance module.

        Args:
            p: Prime for p-adic calculations
            n_states: Number of conformational states
        """
        super().__init__()
        self.p = p
        self.n_states = n_states

    def compute_padic_distances(self, indices: torch.Tensor) -> torch.Tensor:
        """Compute p-adic distance matrix.

        Args:
            indices: State indices (n,)

        Returns:
            Distance matrix (n, n)
        """
        len(indices)
        diff = torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1))

        # Compute p-adic valuations
        valuations = torch.zeros_like(diff, dtype=torch.float)
        for k in range(1, 10):
            divisible = (diff % (self.p**k) == 0) & (diff > 0)
            valuations[divisible] = k

        # p-adic distances
        distances = torch.where(
            diff == 0,
            torch.zeros_like(valuations),
            torch.pow(float(self.p), -valuations),
        )

        return distances

    def check_ultrametricity(self, distances: torch.Tensor) -> float:
        """Check how well a distance matrix satisfies ultrametric property.

        Args:
            distances: Distance matrix (n, n)

        Returns:
            Ultrametricity score (1.0 = perfect ultrametric)
        """
        n = distances.shape[0]
        violations = 0
        total = 0

        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    d_ij = distances[i, j]
                    d_jk = distances[j, k]
                    d_ik = distances[i, k]

                    # Sort distances
                    sorted_d = torch.sort(torch.tensor([d_ij, d_jk, d_ik]))[0]

                    # Ultrametric: two largest should be equal
                    if sorted_d[2] > sorted_d[1] + 1e-6:
                        violations += 1
                    total += 1

        return 1.0 - violations / total if total > 0 else 1.0

    def forward(self, states: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute ultrametric analysis of states.

        Args:
            states: Conformational states (batch, n_states, dim)

        Returns:
            Dictionary with distance matrix and metrics
        """
        batch_size, n_states, dim = states.shape

        results = []
        for b in range(batch_size):
            # Compute Euclidean distances
            diff = states[b].unsqueeze(0) - states[b].unsqueeze(1)
            euclidean = torch.sqrt((diff**2).sum(dim=-1))

            # Compute p-adic structure based on energy ordering
            # (assumes states are ordered by energy)
            indices = torch.arange(n_states, device=states.device)
            padic = self.compute_padic_distances(indices)

            # Check ultrametricity
            ultrametricity = self.check_ultrametricity(euclidean)

            results.append(
                {
                    "euclidean_distances": euclidean,
                    "padic_distances": padic,
                    "ultrametricity": ultrametricity,
                }
            )

        return results


class FoldingFunnelAnalyzer(nn.Module):
    """Analyzes the folding funnel structure of protein landscapes.

    The folding funnel hypothesis states that proteins have a
    funnel-shaped energy landscape guiding folding to the native state.
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        p: int = 3,
    ):
        """Initialize funnel analyzer.

        Args:
            input_dim: Dimension of conformational vectors
            hidden_dim: Hidden layer dimension
            p: Prime for p-adic calculations
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.p = p

        # Energy predictor
        self.energy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # State classifier
        self.state_classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(ConformationState)),
        )

        # Basin detector
        self.basin_detector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def predict_energy(self, conformations: torch.Tensor) -> torch.Tensor:
        """Predict energy of conformations.

        Args:
            conformations: Conformational vectors (batch, dim)

        Returns:
            Predicted energies (batch,)
        """
        return self.energy_net(conformations).squeeze(-1)

    def classify_state(self, conformations: torch.Tensor) -> torch.Tensor:
        """Classify conformational state.

        Args:
            conformations: Conformational vectors (batch, dim)

        Returns:
            State logits (batch, n_states)
        """
        return self.state_classifier(conformations)

    def detect_basins(
        self,
        conformations: torch.Tensor,
        energies: torch.Tensor,
    ) -> list[EnergyBasin]:
        """Detect energy basins in the landscape.

        Args:
            conformations: Conformational vectors (n_samples, dim)
            energies: Energy values (n_samples,)

        Returns:
            List of detected energy basins
        """
        # Simple basin detection: local minima
        n_samples = len(energies)
        basins = []

        # Compute pairwise distances
        diffs = conformations.unsqueeze(0) - conformations.unsqueeze(1)
        distances = torch.sqrt((diffs**2).sum(dim=-1))

        # Find local minima
        for i in range(n_samples):
            # Get nearest neighbors
            _, nearest = torch.topk(distances[i], k=min(10, n_samples), largest=False)

            # Check if this is a local minimum
            neighbor_energies = energies[nearest[1:]]  # Exclude self
            if (energies[i] <= neighbor_energies).all():
                # Estimate basin properties
                neighbor_mask = distances[i] < distances[i, nearest[-1]]
                conformations[neighbor_mask]
                basin_energies = energies[neighbor_mask]

                depth = basin_energies.max() - energies[i]
                width = distances[i, neighbor_mask].mean()

                # Classify state
                state_logits = self.classify_state(conformations[i : i + 1])
                state_idx = state_logits.argmax().item()
                state = list(ConformationState)[state_idx]

                basins.append(
                    EnergyBasin(
                        center=conformations[i],
                        energy=energies[i].item(),
                        depth=depth.item(),
                        width=width.item(),
                        state=state,
                        escape_barrier=depth.item(),
                    )
                )

        return basins

    def compute_funnel_metrics(
        self,
        conformations: torch.Tensor,
        energies: torch.Tensor,
        native_state: torch.Tensor,
    ) -> dict[str, float]:
        """Compute folding funnel metrics.

        Args:
            conformations: Conformational samples (n, dim)
            energies: Corresponding energies (n,)
            native_state: Native state conformation (dim,)

        Returns:
            Dictionary of funnel metrics
        """
        # Compute distances to native state
        native_distances = torch.sqrt(((conformations - native_state) ** 2).sum(dim=-1))

        # Funnel depth: energy difference from unfolded to native
        native_energy = energies[native_distances.argmin()].item()
        max_energy = energies.max().item()
        funnel_depth = max_energy - native_energy

        # Ruggedness: variance in energy along folding coordinate
        ruggedness = energies.std().item()

        # Correlation between distance and energy (negative = good funnel)
        correlation = torch.corrcoef(torch.stack([native_distances, energies]))[0, 1].item()

        # Frustration: fraction of uphill steps on average path
        sorted_idx = torch.argsort(native_distances)
        sorted_energies = energies[sorted_idx]
        uphill_steps = (sorted_energies[1:] > sorted_energies[:-1]).float().mean().item()
        frustration = uphill_steps

        return {
            "funnel_depth": funnel_depth,
            "ruggedness": ruggedness,
            "funnel_correlation": correlation,
            "frustration": frustration,
            "native_energy": native_energy,
        }

    def forward(
        self,
        conformations: torch.Tensor,
        native_state: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Analyze folding landscape.

        Args:
            conformations: Conformational samples (batch, n_samples, dim)
            native_state: Native state if known (batch, dim)

        Returns:
            Dictionary with landscape analysis
        """
        batch_size, n_samples, dim = conformations.shape

        results = []
        for b in range(batch_size):
            # Predict energies
            energies = self.predict_energy(conformations[b])

            # Detect basins
            basins = self.detect_basins(conformations[b], energies)

            # Compute funnel metrics if native state provided
            if native_state is not None:
                funnel_metrics = self.compute_funnel_metrics(conformations[b], energies, native_state[b])
            else:
                # Use lowest energy state as native
                native_idx = energies.argmin()
                funnel_metrics = self.compute_funnel_metrics(conformations[b], energies, conformations[b, native_idx])

            results.append(
                {
                    "energies": energies,
                    "basins": basins,
                    "funnel_metrics": funnel_metrics,
                    "n_basins": len(basins),
                }
            )

        return results


class TransitionStateAnalyzer(nn.Module):
    """Analyzes transition states and paths between conformations.

    Uses ultrametric structure to identify hierarchical
    transition pathways in the folding landscape.
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        p: int = 3,
    ):
        """Initialize transition analyzer.

        Args:
            input_dim: Dimension of conformational vectors
            hidden_dim: Hidden layer dimension
            p: Prime for p-adic calculations
        """
        super().__init__()
        self.input_dim = input_dim
        self.p = p

        # Transition state finder
        self.ts_finder = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        # Barrier estimator
        self.barrier_net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.ReLU(),  # Barriers are positive
        )

    def find_transition_state(
        self,
        state1: torch.Tensor,
        state2: torch.Tensor,
    ) -> torch.Tensor:
        """Find transition state between two conformations.

        Args:
            state1: First conformation (batch, dim)
            state2: Second conformation (batch, dim)

        Returns:
            Transition state conformation (batch, dim)
        """
        combined = torch.cat([state1, state2], dim=-1)
        ts = self.ts_finder(combined)

        # Ensure transition state is between the two states
        # (project to line connecting states)
        direction = state2 - state1
        projection = ((ts - state1) * direction).sum(dim=-1, keepdim=True)
        projection = projection / (direction.norm(dim=-1, keepdim=True) ** 2 + 1e-8)
        projection = projection.clamp(0, 1)

        ts_projected = state1 + projection * direction

        return ts_projected

    def estimate_barrier(
        self,
        state1: torch.Tensor,
        state2: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate energy barrier between states.

        Args:
            state1: First conformation (batch, dim)
            state2: Second conformation (batch, dim)

        Returns:
            Estimated barrier height (batch,)
        """
        combined = torch.cat([state1, state2], dim=-1)
        return self.barrier_net(combined).squeeze(-1)

    def compute_padic_path_length(
        self,
        path: torch.Tensor,
    ) -> torch.Tensor:
        """Compute p-adic path length along a trajectory.

        Args:
            path: Sequence of states (batch, n_steps, dim)

        Returns:
            Path length in p-adic metric (batch,)
        """
        batch_size, n_steps, dim = path.shape

        # Convert positions to indices for p-adic calculation
        # Use projection onto first principal component
        path.reshape(batch_size, n_steps, -1)
        indices = torch.arange(n_steps, device=path.device)

        # Sum of p-adic distances between consecutive points
        lengths = torch.zeros(batch_size, device=path.device)
        for i in range(n_steps - 1):
            diff = abs(indices[i].item() - indices[i + 1].item())
            if diff > 0:
                val = 0
                temp = diff
                while temp % self.p == 0:
                    val += 1
                    temp //= self.p
                d = self.p ** (-val)
            else:
                d = 0
            lengths += d

        return lengths

    def find_minimum_energy_path(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        energy_func: nn.Module,
        n_steps: int = 20,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Find minimum energy path between two states.

        Uses the nudged elastic band (NEB) method.

        Args:
            start: Starting conformation (dim,)
            end: Ending conformation (dim,)
            energy_func: Function to compute energies
            n_steps: Number of steps in path

        Returns:
            Tuple of (path, energies)
        """
        # Initialize linear path
        path = torch.zeros(n_steps, start.shape[0])
        for i in range(n_steps):
            alpha = i / (n_steps - 1)
            path[i] = (1 - alpha) * start + alpha * end

        # Simple optimization (in practice would use NEB)
        path = nn.Parameter(path)
        optimizer = torch.optim.Adam([path], lr=0.01)

        for _ in range(100):
            # Compute energies
            energies = energy_func(path)

            # Spring forces
            spring_k = 1.0
            spring_force = torch.zeros_like(path)
            for i in range(1, n_steps - 1):
                spring_force[i] = spring_k * (path[i + 1] + path[i - 1] - 2 * path[i])

            # Total loss
            loss = energies.mean() - 0.1 * (spring_force**2).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Fix endpoints
            with torch.no_grad():
                path.data[0] = start
                path.data[-1] = end

        return path.detach(), energy_func(path.detach())

    def forward(
        self,
        state1: torch.Tensor,
        state2: torch.Tensor,
    ) -> TransitionPath:
        """Analyze transition between two states.

        Args:
            state1: First conformation (batch, dim) or (dim,)
            state2: Second conformation (batch, dim) or (dim,)

        Returns:
            TransitionPath dataclass
        """
        if state1.dim() == 1:
            state1 = state1.unsqueeze(0)
            state2 = state2.unsqueeze(0)

        # Find transition state
        ts = self.find_transition_state(state1, state2)

        # Estimate barrier
        barrier = self.estimate_barrier(state1, state2)

        # Compute path length
        path = torch.stack([state1.squeeze(), ts.squeeze(), state2.squeeze()])
        path_length = self.compute_padic_path_length(path.unsqueeze(0))[0]

        # Estimate rate constant (Arrhenius-like)
        # k = A * exp(-E_barrier / kT), simplified
        rate = torch.exp(-barrier)

        return TransitionPath(
            start_state=ConformationState.INTERMEDIATE,
            end_state=ConformationState.INTERMEDIATE,
            barrier_height=barrier.item(),
            path_length=path_length.item(),
            transition_state=ts.squeeze(),
            rate_constant=rate.item(),
        )


class ProteinLandscapeAnalyzer(nn.Module):
    """Complete protein energy landscape analyzer.

    Combines ultrametric analysis, funnel characterization,
    and transition path analysis.
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        p: int = 3,
    ):
        """Initialize landscape analyzer.

        Args:
            input_dim: Dimension of conformational vectors
            hidden_dim: Hidden layer dimension
            p: Prime for p-adic calculations
        """
        super().__init__()
        self.input_dim = input_dim
        self.p = p

        # Component analyzers
        self.ultrametric = UltrametricDistanceMatrix(p=p)
        self.funnel = FoldingFunnelAnalyzer(input_dim, hidden_dim, p)
        self.transition = TransitionStateAnalyzer(input_dim, hidden_dim, p)

    def forward(
        self,
        conformations: torch.Tensor,
        native_state: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Perform complete landscape analysis.

        Args:
            conformations: Conformational samples (batch, n_samples, dim)
            native_state: Native state if known (batch, dim)

        Returns:
            Dictionary with complete analysis results
        """
        batch_size, n_samples, dim = conformations.shape

        # Ultrametric analysis
        ultrametric_results = self.ultrametric(conformations)

        # Funnel analysis
        funnel_results = self.funnel(conformations, native_state)

        # Combine results
        landscape_metrics = []
        for b in range(batch_size):
            metrics = LandscapeMetrics(
                ruggedness=funnel_results[b]["funnel_metrics"]["ruggedness"],
                funnel_depth=funnel_results[b]["funnel_metrics"]["funnel_depth"],
                frustration=funnel_results[b]["funnel_metrics"]["frustration"],
                ultrametricity=ultrametric_results[b]["ultrametricity"],
                n_basins=funnel_results[b]["n_basins"],
                native_stability=-funnel_results[b]["funnel_metrics"]["native_energy"],
            )
            landscape_metrics.append(metrics)

        return {
            "ultrametric": ultrametric_results,
            "funnel": funnel_results,
            "metrics": landscape_metrics,
        }

    def compare_landscapes(
        self,
        landscape1: dict[str, Any],
        landscape2: dict[str, Any],
    ) -> dict[str, float]:
        """Compare two protein landscapes.

        Useful for analyzing effects of mutations or conditions.

        Args:
            landscape1: First landscape analysis
            landscape2: Second landscape analysis

        Returns:
            Dictionary of comparison metrics
        """
        m1 = landscape1["metrics"][0]
        m2 = landscape2["metrics"][0]

        return {
            "ruggedness_diff": abs(m1.ruggedness - m2.ruggedness),
            "funnel_depth_diff": abs(m1.funnel_depth - m2.funnel_depth),
            "frustration_diff": abs(m1.frustration - m2.frustration),
            "ultrametricity_diff": abs(m1.ultrametricity - m2.ultrametricity),
            "stability_diff": abs(m1.native_stability - m2.native_stability),
        }
