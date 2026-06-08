# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Statistical Physics module for biological systems.

This module implements physics-inspired methods for analyzing protein
folding landscapes, sampling conformational states, and extracting
hierarchical structure from biological data.

Key features:
- Spin glass models for protein energy landscapes
- Replica exchange (parallel tempering) for enhanced sampling
- Ultrametric tree extraction via replica symmetry breaking
- P-adic connections to ultrametric structure

References:
- Hopfield (1982): Neural networks and physical systems
- Parisi (1983): Replica symmetry breaking in spin glasses
- Swendsen & Wang (1986): Replica Monte Carlo simulation
- Rammal et al. (1986): Ultrametricity for physicists
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.padic_math import padic_valuation


@dataclass
class EnergyState:
    """Represents a state in the energy landscape."""

    configuration: torch.Tensor
    energy: float
    temperature: float
    metadata: dict = field(default_factory=dict)


class SpinGlassLandscape(nn.Module):
    """Spin glass model for protein energy landscapes.

    Models the energy landscape of a biological system (e.g., protein)
    as a spin glass with complex interactions between sites.

    The Hamiltonian is:
        H = -Σᵢⱼ Jᵢⱼ sᵢ sⱼ - Σᵢ hᵢ sᵢ

    where J is the coupling matrix and h is the external field.

    In the context of proteins:
    - Spins represent residue states (conformations, rotamers)
    - Couplings encode pairwise interactions (contacts, correlations)
    - External field encodes site-specific preferences
    """

    def __init__(
        self,
        n_sites: int,
        n_states: int = 2,
        coupling_type: str = "gaussian",
        coupling_scale: float = 1.0,
        field_scale: float = 0.1,
        sparse_fraction: float = 1.0,
    ):
        """Initialize spin glass landscape.

        Args:
            n_sites: Number of sites (e.g., residues)
            n_states: Number of states per site (2 for Ising, >2 for Potts)
            coupling_type: Type of couplings ('gaussian', 'uniform', 'hopfield')
            coupling_scale: Scale of coupling strengths
            field_scale: Scale of external field
            sparse_fraction: Fraction of non-zero couplings
        """
        super().__init__()
        self.n_sites = n_sites
        self.n_states = n_states
        self.coupling_type = coupling_type
        self.coupling_scale = coupling_scale
        self.field_scale = field_scale
        self.sparse_fraction = sparse_fraction

        # Initialize couplings
        self._init_couplings()

        # External field
        self.field = nn.Parameter(torch.randn(n_sites, n_states) * field_scale, requires_grad=False)

    def _init_couplings(self):
        """Initialize coupling matrix based on type."""
        n = self.n_sites
        q = self.n_states

        if self.coupling_type == "gaussian":
            # Random Gaussian couplings (Sherrington-Kirkpatrick model)
            J = torch.randn(n, n, q, q) * self.coupling_scale / math.sqrt(n)

        elif self.coupling_type == "uniform":
            # Uniform random couplings
            J = (2 * torch.rand(n, n, q, q) - 1) * self.coupling_scale

        elif self.coupling_type == "hopfield":
            # Hopfield-like ferromagnetic couplings
            J = torch.ones(n, n, q, q) * self.coupling_scale / n

        else:
            raise ValueError(f"Unknown coupling type: {self.coupling_type}")

        # Make symmetric
        J = (J + J.permute(1, 0, 3, 2)) / 2

        # Zero diagonal (no self-interaction)
        for i in range(n):
            J[i, i] = 0

        # Apply sparsity
        if self.sparse_fraction < 1.0:
            mask = torch.rand(n, n) < self.sparse_fraction
            mask = mask | mask.T  # Symmetric
            J = J * mask.unsqueeze(-1).unsqueeze(-1)

        self.couplings = nn.Parameter(J, requires_grad=False)

    def energy(self, configuration: torch.Tensor) -> torch.Tensor:
        """Compute energy of configuration(s).

        Args:
            configuration: State assignments, shape (..., n_sites) with values in [0, n_states)

        Returns:
            Energy, shape (...)
        """
        batch_shape = configuration.shape[:-1]
        config_flat = configuration.reshape(-1, self.n_sites)
        batch_size = config_flat.shape[0]

        # Convert to one-hot
        one_hot = F.one_hot(config_flat.long(), self.n_states).float()

        # Pairwise interaction energy: -Σᵢⱼ Jᵢⱼ[sᵢ,sⱼ]
        # Shape: (batch, n_sites, n_states) @ (n_sites, n_sites, n_states, n_states) @ (batch, n_sites, n_states)
        interaction_energy = torch.zeros(batch_size, device=configuration.device)

        for b in range(batch_size):
            for i in range(self.n_sites):
                for j in range(i + 1, self.n_sites):
                    si, sj = config_flat[b, i].long(), config_flat[b, j].long()
                    interaction_energy[b] -= self.couplings[i, j, si, sj]

        # Field energy: -Σᵢ hᵢ[sᵢ]
        field_energy = -(one_hot * self.field).sum(dim=(-1, -2))

        total_energy = interaction_energy + field_energy
        return total_energy.reshape(batch_shape)

    def energy_vectorized(self, configuration: torch.Tensor) -> torch.Tensor:
        """Vectorized energy computation for batched configurations.

        Args:
            configuration: Shape (batch, n_sites)

        Returns:
            Energy, shape (batch,)
        """
        batch_size = configuration.shape[0]
        one_hot = F.one_hot(configuration.long(), self.n_states).float()

        # Interaction: sum over pairs
        # Expand for batch einsum
        # J: (n_sites, n_sites, n_states, n_states)
        # one_hot: (batch, n_sites, n_states)

        # Use einsum for efficient computation
        # E_int = -0.5 * Σᵢⱼ Jᵢⱼₐᵦ sᵢₐ sⱼᵦ
        interaction = torch.einsum(
            "bi,bj,ijab,bia,bjb->b",
            torch.ones(batch_size, self.n_sites, device=configuration.device),
            torch.ones(batch_size, self.n_sites, device=configuration.device),
            self.couplings,
            one_hot,
            one_hot,
        )
        interaction_energy = -0.5 * interaction

        # Field energy
        field_energy = -(one_hot * self.field).sum(dim=(-1, -2))

        return interaction_energy + field_energy

    def local_field(self, configuration: torch.Tensor, site: int) -> torch.Tensor:
        """Compute local field at a site given other spins.

        Args:
            configuration: Current configuration, shape (..., n_sites)
            site: Site index

        Returns:
            Local field for each state, shape (..., n_states)
        """
        batch_shape = configuration.shape[:-1]
        config_flat = configuration.reshape(-1, self.n_sites)

        # Contribution from other spins
        local = self.field[site].clone().unsqueeze(0).expand(len(config_flat), -1)

        for j in range(self.n_sites):
            if j != site:
                sj = config_flat[:, j].long()
                local = local + self.couplings[site, j, :, sj].T

        return local.reshape(*batch_shape, self.n_states)

    def forward(self, configuration: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Compute Boltzmann probability of configuration.

        Args:
            configuration: Shape (..., n_sites)
            temperature: Temperature (default 1.0)

        Returns:
            Log probability (unnormalized)
        """
        energy = self.energy_vectorized(configuration)
        return -energy / temperature


class ReplicaExchange:
    """Replica Exchange Monte Carlo (Parallel Tempering).

    Runs multiple replicas at different temperatures and exchanges
    configurations between adjacent temperature levels to enhance
    sampling of multi-modal distributions.

    The exchange probability is:
        P(swap) = min(1, exp(ΔβΔE))

    where Δβ = 1/T₂ - 1/T₁ and ΔE = E₂ - E₁.
    """

    def __init__(
        self,
        n_replicas: int = 8,
        temp_min: float = 0.1,
        temp_max: float = 10.0,
        n_sweeps: int = 100,
        exchange_frequency: int = 10,
        temp_schedule: str = "geometric",
    ):
        """Initialize replica exchange sampler.

        Args:
            n_replicas: Number of replicas at different temperatures
            temp_min: Minimum temperature
            temp_max: Maximum temperature
            n_sweeps: Monte Carlo sweeps between exchanges
            exchange_frequency: How often to attempt exchanges
            temp_schedule: Temperature schedule ('geometric', 'linear')
        """
        self.n_replicas = n_replicas
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.n_sweeps = n_sweeps
        self.exchange_frequency = exchange_frequency
        self.temp_schedule = temp_schedule

        # Create temperature ladder
        self.temperatures = self._create_temp_ladder()

    def _create_temp_ladder(self) -> torch.Tensor:
        """Create temperature ladder."""
        if self.temp_schedule == "geometric":
            # Geometric spacing (more temperatures at low T)
            ratio = (self.temp_max / self.temp_min) ** (1 / (self.n_replicas - 1))
            temps = [self.temp_min * (ratio**i) for i in range(self.n_replicas)]

        elif self.temp_schedule == "linear":
            temps = torch.linspace(self.temp_min, self.temp_max, self.n_replicas).tolist()

        else:
            raise ValueError(f"Unknown schedule: {self.temp_schedule}")

        return torch.tensor(temps)

    def _metropolis_sweep(
        self,
        landscape: SpinGlassLandscape,
        configuration: torch.Tensor,
        temperature: float,
    ) -> tuple[torch.Tensor, int]:
        """Perform one Metropolis sweep.

        Args:
            landscape: Energy landscape
            configuration: Current configuration
            temperature: Temperature

        Returns:
            New configuration, number of accepted moves
        """
        config = configuration.clone()
        n_accepted = 0
        n_sites = landscape.n_sites
        n_states = landscape.n_states

        for site in range(n_sites):
            old_state = config[site].item()
            current_energy = landscape.energy(config.unsqueeze(0)).item()

            # Propose new state
            new_state = torch.randint(n_states, (1,)).item()
            if new_state == old_state:
                continue

            # Compute energy change
            config[site] = new_state
            new_energy = landscape.energy(config.unsqueeze(0)).item()
            delta_E = new_energy - current_energy

            # Accept/reject
            if delta_E < 0 or torch.rand(1).item() < math.exp(-delta_E / temperature):
                n_accepted += 1
            else:
                config[site] = old_state

        return config, n_accepted

    def _attempt_exchange(
        self,
        landscape: SpinGlassLandscape,
        configs: list[torch.Tensor],
        i: int,
        j: int,
    ) -> bool:
        """Attempt exchange between replicas i and j.

        Args:
            landscape: Energy landscape
            configs: List of configurations for each replica
            i, j: Replica indices

        Returns:
            Whether exchange was accepted
        """
        Ti, Tj = self.temperatures[i].item(), self.temperatures[j].item()
        Ei = landscape.energy(configs[i].unsqueeze(0)).item()
        Ej = landscape.energy(configs[j].unsqueeze(0)).item()

        # Exchange criterion
        delta = (1 / Ti - 1 / Tj) * (Ej - Ei)

        if delta > 0 or torch.rand(1).item() < math.exp(delta):
            # Accept exchange
            configs[i], configs[j] = configs[j].clone(), configs[i].clone()
            return True
        return False

    def sample(
        self,
        landscape: SpinGlassLandscape,
        n_samples: int = 1000,
        initial_configs: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run replica exchange sampling.

        Args:
            landscape: Energy landscape to sample
            n_samples: Number of samples to collect from lowest T replica
            initial_configs: Initial configurations for each replica

        Returns:
            Dict with samples, energies, and diagnostics
        """
        n_sites = landscape.n_sites
        n_states = landscape.n_states

        # Initialize configurations
        if initial_configs is None:
            configs = [torch.randint(n_states, (n_sites,)) for _ in range(self.n_replicas)]
        else:
            configs = [c.clone() for c in initial_configs]

        # Storage for samples
        samples = []
        energies = []

        # Sampling loop
        sample_interval = max(1, self.n_sweeps * self.exchange_frequency // n_samples)
        n_exchanges_accepted = 0
        n_exchanges_attempted = 0

        for iteration in range(n_samples * sample_interval):
            # Metropolis sweeps for each replica
            for r in range(self.n_replicas):
                configs[r], _ = self._metropolis_sweep(landscape, configs[r], self.temperatures[r].item())

            # Attempt exchanges
            if iteration % self.exchange_frequency == 0:
                for r in range(self.n_replicas - 1):
                    n_exchanges_attempted += 1
                    if self._attempt_exchange(landscape, configs, r, r + 1):
                        n_exchanges_accepted += 1

            # Collect samples from lowest temperature replica
            if iteration % sample_interval == 0:
                samples.append(configs[0].clone())
                energies.append(landscape.energy(configs[0].unsqueeze(0)).item())

        # Compute exchange rate
        exchange_rate = n_exchanges_accepted / n_exchanges_attempted if n_exchanges_attempted > 0 else 0.0

        return {
            "samples": torch.stack(samples),
            "energies": torch.tensor(energies),
            "exchange_rate": exchange_rate,
            "temperatures": self.temperatures,
            "final_configs": torch.stack(configs),
        }


class UltrametricTreeExtractor:
    """Extract ultrametric tree structure from distance/similarity data.

    Uses the connection between ultrametricity and p-adic structure
    to build hierarchical trees from biological data.

    An ultrametric satisfies the strong triangle inequality:
        d(x,z) ≤ max(d(x,y), d(y,z))

    This is equivalent to the p-adic metric and corresponds to
    hierarchical clustering (UPGMA, complete linkage).
    """

    def __init__(
        self,
        linkage: str = "average",
        prime: int = 3,
        threshold: float = 1e-6,
    ):
        """Initialize ultrametric extractor.

        Args:
            linkage: Linkage type ('average', 'complete', 'single')
            prime: Prime for p-adic interpretation
            threshold: Threshold for ultrametric violations
        """
        self.linkage = linkage
        self.prime = prime
        self.threshold = threshold

    def check_ultrametricity(self, distance_matrix: torch.Tensor) -> tuple[bool, float]:
        """Check if distance matrix is ultrametric.

        Args:
            distance_matrix: Symmetric distance matrix

        Returns:
            Tuple of (is_ultrametric, max_violation)
        """
        n = distance_matrix.shape[0]
        max_violation = 0.0

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if i in (j, k) or j == k:
                        continue

                    dij = distance_matrix[i, j].item()
                    djk = distance_matrix[j, k].item()
                    dik = distance_matrix[i, k].item()

                    # Check strong triangle inequality
                    violation = dik - max(dij, djk)
                    if violation > max_violation:
                        max_violation = violation

        return max_violation <= self.threshold, max_violation

    def make_ultrametric(self, distance_matrix: torch.Tensor) -> torch.Tensor:
        """Convert distance matrix to ultrametric via hierarchical clustering.

        Args:
            distance_matrix: Symmetric distance matrix

        Returns:
            Ultrametric distance matrix (cophenetic distances)
        """
        n = distance_matrix.shape[0]
        D = distance_matrix.clone()

        # Build tree via agglomerative clustering
        cluster_members = [[i] for i in range(n)]
        active_clusters = list(range(n))

        cophenetic = torch.zeros_like(D)

        while len(active_clusters) > 1:
            # Find closest pair of clusters
            min_dist = float("inf")
            best_i, best_j = None, None

            for ii, ci in enumerate(active_clusters):
                for jj, cj in enumerate(active_clusters[ii + 1 :], ii + 1):
                    dist = self._cluster_distance(D, cluster_members[ci], cluster_members[cj])
                    if dist < min_dist:
                        min_dist = dist
                        best_i, best_j = ii, jj

            # Merge clusters
            ci, cj = active_clusters[best_i], active_clusters[best_j]

            # Record cophenetic distances
            for mi in cluster_members[ci]:
                for mj in cluster_members[cj]:
                    cophenetic[mi, mj] = min_dist
                    cophenetic[mj, mi] = min_dist

            # Create new cluster
            new_cluster = cluster_members[ci] + cluster_members[cj]
            cluster_members.append(new_cluster)
            new_idx = len(cluster_members) - 1

            # Update active clusters
            active_clusters.remove(ci)
            active_clusters.remove(cj)
            active_clusters.append(new_idx)

        return cophenetic

    def _cluster_distance(
        self,
        D: torch.Tensor,
        members_i: list[int],
        members_j: list[int],
    ) -> float:
        """Compute distance between clusters based on linkage."""
        distances = []
        for i in members_i:
            for j in members_j:
                distances.append(D[i, j].item())

        if self.linkage == "average":
            return sum(distances) / len(distances)
        elif self.linkage == "complete":
            return max(distances)
        elif self.linkage == "single":
            return min(distances)
        else:
            raise ValueError(f"Unknown linkage: {self.linkage}")

    def extract_tree(self, distance_matrix: torch.Tensor) -> dict[str, torch.Tensor | list]:
        """Extract ultrametric tree from distance matrix.

        Args:
            distance_matrix: Symmetric distance matrix

        Returns:
            Dict with tree structure, heights, and p-adic valuations
        """
        # Make ultrametric if needed
        is_ultra, violation = self.check_ultrametricity(distance_matrix)

        ultrametric_D = self.make_ultrametric(distance_matrix) if not is_ultra else distance_matrix.clone()

        # Extract tree structure
        n = distance_matrix.shape[0]
        merge_order = []
        heights = []

        # Find unique distances (merge heights)
        unique_dists = torch.unique(ultrametric_D)
        unique_dists = unique_dists[unique_dists > 0]
        unique_dists, _ = torch.sort(unique_dists)

        # Build Newick-style tree
        current_labels = [str(i) for i in range(n)]

        for h in unique_dists:
            # Find pairs at this distance
            mask = (ultrametric_D == h) & (torch.triu(torch.ones(n, n), diagonal=1).bool())
            pairs = torch.nonzero(mask)

            for pair in pairs:
                i, j = pair[0].item(), pair[1].item()
                if current_labels[i] != current_labels[j]:
                    # Merge
                    new_label = f"({current_labels[i]},{current_labels[j]})"
                    merge_order.append((i, j, h.item()))
                    heights.append(h.item())

                    # Update labels
                    for k in range(n):
                        if current_labels[k] == current_labels[i] or current_labels[k] == current_labels[j]:
                            current_labels[k] = new_label

        # Compute p-adic valuations of heights using centralized function
        valuations = [self._padic_valuation_float(h) for h in heights]

        return {
            "ultrametric_distance": ultrametric_D,
            "merge_order": merge_order,
            "heights": torch.tensor(heights),
            "padic_valuations": valuations,
            "was_ultrametric": is_ultra,
            "max_violation": violation,
        }

    def _padic_valuation_float(self, x: float) -> int:
        """Compute approximate p-adic valuation for float values.

        Uses centralized padic_valuation after discretizing the float.
        """
        if x == 0:
            return float("inf")

        # Discretize to find power of prime (scale to integer)
        x_int = int(round(x * 1000))

        if x_int == 0:
            return float("inf")

        return padic_valuation(x_int, self.prime)


class ParisiOverlapAnalyzer:
    """Analyze replica overlap distribution (Parisi order parameter).

    In spin glasses, the overlap q = (1/N)Σᵢ sᵢᵅ sᵢᵝ between replicas
    reveals the structure of the free energy landscape.

    - Single peak at q=0: Paramagnetic phase
    - Single peak at q≠0: Ferromagnetic phase
    - Multiple peaks: Spin glass phase (replica symmetry breaking)
    """

    def __init__(self, n_bins: int = 50):
        """Initialize overlap analyzer.

        Args:
            n_bins: Number of bins for overlap histogram
        """
        self.n_bins = n_bins

    def compute_overlap(self, config1: torch.Tensor, config2: torch.Tensor) -> float:
        """Compute overlap between two configurations.

        For Ising spins (+1/-1):
            q = (1/N) Σᵢ sᵢ⁽¹⁾ sᵢ⁽²⁾

        For Potts spins (0,1,...,q-1):
            q = (1/N) Σᵢ δ(sᵢ⁽¹⁾, sᵢ⁽²⁾) - 1/q

        Args:
            config1, config2: Spin configurations

        Returns:
            Overlap value in [-1, 1]
        """
        # Check if Ising-like (values are 0 or 1)
        if config1.max() <= 1:
            # Convert to +1/-1
            s1 = 2 * config1.float() - 1
            s2 = 2 * config2.float() - 1
            return (s1 * s2).mean().item()
        else:
            # Potts overlap
            q = config1.max().item() + 1
            matches = (config1 == config2).float().mean().item()
            return (matches - 1 / q) / (1 - 1 / q)

    def overlap_distribution(self, samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute overlap distribution from samples.

        Args:
            samples: Shape (n_samples, n_sites)

        Returns:
            Tuple of (bin_centers, histogram)
        """
        n_samples = len(samples)
        overlaps = []

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                q = self.compute_overlap(samples[i], samples[j])
                overlaps.append(q)

        overlaps = torch.tensor(overlaps)

        # Create histogram
        hist, bin_edges = torch.histogram(overlaps, bins=self.n_bins, range=(-1.0, 1.0))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Normalize
        hist = hist.float() / hist.sum()

        return bin_centers, hist

    def analyze_phase(self, samples: torch.Tensor) -> dict[str, float | str | torch.Tensor]:
        """Analyze the phase from overlap distribution.

        Args:
            samples: Shape (n_samples, n_sites)

        Returns:
            Dict with phase analysis results
        """
        bin_centers, hist = self.overlap_distribution(samples)

        # Compute statistics
        mean_overlap = (bin_centers * hist).sum().item()
        var_overlap = ((bin_centers - mean_overlap) ** 2 * hist).sum().item()

        # Find peaks
        peaks = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:
                peaks.append((bin_centers[i].item(), hist[i].item()))

        # Classify phase
        if len(peaks) == 1:
            phase = "paramagnetic" if abs(peaks[0][0]) < 0.1 else "ferromagnetic"
        elif len(peaks) == 2 and abs(peaks[0][0] + peaks[1][0]) < 0.1:
            phase = "ferromagnetic (symmetric)"
        else:
            phase = "spin_glass"

        return {
            "phase": phase,
            "mean_overlap": mean_overlap,
            "variance": var_overlap,
            "peaks": peaks,
            "bin_centers": bin_centers,
            "histogram": hist,
        }


class BoltzmannMachine(nn.Module):
    """Restricted Boltzmann Machine with p-adic structure.

    RBM with visible and hidden layers, where hidden unit
    connectivity reflects p-adic hierarchical structure.

    Energy: E(v,h) = -a'v - b'h - v'Wh
    """

    def __init__(
        self,
        n_visible: int,
        n_hidden: int,
        use_padic_structure: bool = True,
        prime: int = 3,
    ):
        """Initialize RBM.

        Args:
            n_visible: Number of visible units
            n_hidden: Number of hidden units
            use_padic_structure: Apply p-adic sparsity to weights
            prime: Prime for p-adic structure
        """
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.prime = prime
        self.use_padic_structure = use_padic_structure

        # Parameters
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.a = nn.Parameter(torch.zeros(n_visible))  # Visible bias
        self.b = nn.Parameter(torch.zeros(n_hidden))  # Hidden bias

        if use_padic_structure:
            self._apply_padic_structure()

    def _apply_padic_structure(self):
        """Apply p-adic hierarchical structure to weights."""
        with torch.no_grad():
            for i in range(self.n_visible):
                for j in range(self.n_hidden):
                    # Connection strength based on p-adic distance
                    v = self._padic_valuation(abs(i - j))
                    scale = self.prime ** (-v)
                    self.W[i, j] *= scale

    def _padic_valuation(self, n: int) -> int:
        """Compute p-adic valuation using centralized implementation.

        Uses centralized padic_valuation from src.core.padic_math.
        Returns capped value of 5 for n=0 to maintain weight scaling behavior.
        """
        if n == 0:
            return 5  # Cap infinite valuation for weight scaling
        return padic_valuation(n, self.prime)

    def energy(self, v: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Compute energy of visible-hidden configuration.

        Args:
            v: Visible units, shape (batch, n_visible)
            h: Hidden units, shape (batch, n_hidden)

        Returns:
            Energy, shape (batch,)
        """
        return -(v @ self.a) - (h @ self.b) - (v @ self.W @ h.T).diag()

    def sample_hidden(self, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample hidden units given visible.

        Args:
            v: Visible activations, shape (batch, n_visible)

        Returns:
            Tuple of (hidden probabilities, hidden samples)
        """
        activation = v @ self.W + self.b
        p_h = torch.sigmoid(activation)
        h = torch.bernoulli(p_h)
        return p_h, h

    def sample_visible(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample visible units given hidden.

        Args:
            h: Hidden activations, shape (batch, n_hidden)

        Returns:
            Tuple of (visible probabilities, visible samples)
        """
        activation = h @ self.W.T + self.a
        p_v = torch.sigmoid(activation)
        v = torch.bernoulli(p_v)
        return p_v, v

    def free_energy(self, v: torch.Tensor) -> torch.Tensor:
        """Compute free energy F(v) = -log Σₕ exp(-E(v,h)).

        Args:
            v: Visible units, shape (batch, n_visible)

        Returns:
            Free energy, shape (batch,)
        """
        vbias_term = v @ self.a
        hidden_term = (v @ self.W + self.b).clamp(-80, 80)
        hidden_term = F.softplus(hidden_term).sum(dim=-1)
        return -vbias_term - hidden_term

    def contrastive_divergence(self, v: torch.Tensor, k: int = 1) -> dict[str, torch.Tensor]:
        """Compute contrastive divergence gradients.

        Args:
            v: Visible data, shape (batch, n_visible)
            k: Number of Gibbs sampling steps

        Returns:
            Dict with gradients for W, a, b
        """
        batch_size = v.shape[0]

        # Positive phase
        p_h_pos, h_pos = self.sample_hidden(v)

        # Negative phase (k-step Gibbs)
        h_neg = h_pos.clone()
        for _ in range(k):
            p_v_neg, v_neg = self.sample_visible(h_neg)
            p_h_neg, h_neg = self.sample_hidden(v_neg)

        # Gradients
        grad_W = (v.T @ p_h_pos - v_neg.T @ p_h_neg) / batch_size
        grad_a = (v - v_neg).mean(dim=0)
        grad_b = (p_h_pos - p_h_neg).mean(dim=0)

        return {"W": grad_W, "a": grad_a, "b": grad_b}

    def forward(self, v: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """Reconstruct visible units.

        Args:
            v: Input visible units
            n_samples: Number of Gibbs steps

        Returns:
            Reconstructed visible units
        """
        _, h = self.sample_hidden(v)

        for _ in range(n_samples - 1):
            p_v, v_sample = self.sample_visible(h)
            _, h = self.sample_hidden(v_sample)

        p_v, _ = self.sample_visible(h)
        return p_v
