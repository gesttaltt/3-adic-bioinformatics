# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Vaccine Optimization using Spin Glass Energy Landscapes.

Uses simulated annealing and replica exchange on a multi-objective
energy landscape to find optimal vaccine candidates that:
1. Minimize immune escape across variants
2. Balance manufacturing feasibility
3. Avoid autoimmune risk (Goldilocks zone)
4. Maximize breadth of protection

The energy function combines multiple objectives into a single
landscape that can be explored using physics-inspired sampling.

Mathematical Framework:
  E(vaccine) = E_escape + λ₁ E_manufacturing + λ₂ E_autoimmune + λ₃ E_breadth

  Ground state = optimal vaccine minimizing total energy
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.padic_math import compute_goldilocks_score
from src.physics.statistical_physics import (
    ReplicaExchange,
)


class ObjectiveType(Enum):
    """Types of optimization objectives."""

    ESCAPE = "escape"
    MANUFACTURING = "manufacturing"
    AUTOIMMUNE = "autoimmune"
    BREADTH = "breadth"
    STABILITY = "stability"


@dataclass
class VaccineCandidate:
    """Represents a vaccine candidate.

    Attributes:
        sequence: Amino acid sequence or encoding
        epitopes: Target epitope positions
        energy: Total energy (lower is better)
        component_energies: Per-objective energies
        escape_risk: Escape probability across variants
        autoimmune_score: Goldilocks zone score
        breadth_score: Cross-variant protection
        metadata: Additional information
    """

    sequence: torch.Tensor
    epitopes: list[tuple[int, int]] = field(default_factory=list)
    energy: float = float("inf")
    component_energies: dict[ObjectiveType, float] = field(default_factory=dict)
    escape_risk: float = 0.0
    autoimmune_score: float = 0.0
    breadth_score: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_amino_acids(self, vocab: list[str] | None = None) -> str:
        """Convert sequence tensor to amino acid string."""
        if vocab is None:
            vocab = list("ACDEFGHIKLMNPQRSTVWY-")
        return "".join(vocab[i] for i in self.sequence.long().tolist())


@dataclass
class VaccineOptimizerConfig:
    """Configuration for vaccine optimizer.

    Attributes:
        n_sites: Number of sites in vaccine sequence
        n_states: Number of possible states per site (amino acids)
        n_replicas: Number of replica exchange replicas
        temp_min: Minimum temperature
        temp_max: Maximum temperature
        n_iterations: Number of optimization iterations
        exchange_frequency: How often to attempt replica exchanges
        escape_weight: Weight for escape risk objective
        manufacturing_weight: Weight for manufacturing cost
        autoimmune_weight: Weight for autoimmune risk
        breadth_weight: Weight for breadth of protection
        goldilocks_center: Center of Goldilocks zone
        goldilocks_width: Width of Goldilocks zone
        n_candidates: Number of candidates to return
    """

    n_sites: int = 100
    n_states: int = 21  # 20 amino acids + gap
    n_replicas: int = 8
    temp_min: float = 0.1
    temp_max: float = 10.0
    n_iterations: int = 1000
    exchange_frequency: int = 10
    escape_weight: float = 1.0
    manufacturing_weight: float = 0.3
    autoimmune_weight: float = 0.5
    breadth_weight: float = 0.4
    goldilocks_center: float = 0.5
    goldilocks_width: float = 0.15
    n_candidates: int = 10


@dataclass
class OptimizationResult:
    """Results from vaccine optimization.

    Attributes:
        best_candidate: Optimal vaccine candidate
        all_candidates: Top N candidates
        optimization_trajectory: Energy over iterations
        acceptance_rates: MCMC acceptance rates
        exchange_rates: Replica exchange rates
        pareto_front: Pareto-optimal candidates
    """

    best_candidate: VaccineCandidate
    all_candidates: list[VaccineCandidate]
    optimization_trajectory: list[float]
    acceptance_rates: dict[float, float]
    exchange_rates: list[float]
    pareto_front: list[VaccineCandidate] = field(default_factory=list)


class ImmunogenicityLandscape(nn.Module):
    """Multi-objective energy landscape for vaccine optimization.

    Combines multiple fitness criteria into a single energy function
    that can be minimized via simulated annealing.
    """

    def __init__(
        self,
        config: VaccineOptimizerConfig,
        variant_profiles: torch.Tensor | None = None,
        escape_predictor: nn.Module | None = None,
    ):
        """Initialize immunogenicity landscape.

        Args:
            config: Optimizer configuration
            variant_profiles: Known variant mutation profiles
            escape_predictor: Model predicting escape probability
        """
        super().__init__()
        self.config = config

        # Variant profiles for breadth calculation
        if variant_profiles is not None:
            self.register_buffer("variant_profiles", variant_profiles)
        else:
            # Default: random variants for testing
            self.register_buffer(
                "variant_profiles",
                torch.randint(config.n_states, (10, config.n_sites)),
            )

        # Escape predictor (placeholder if not provided)
        self.escape_predictor = escape_predictor

        # Manufacturing cost model (based on amino acid properties)
        # Some amino acids are harder to synthesize
        self.manufacturing_costs = nn.Parameter(torch.rand(config.n_states) * 0.5, requires_grad=False)

        # Position-specific conservation (high = important for stability)
        self.conservation = nn.Parameter(torch.rand(config.n_sites), requires_grad=False)

        # Known immunogenic positions
        self.immunogenic_positions = nn.Parameter(torch.zeros(config.n_sites), requires_grad=False)

    def energy(self, sequence: torch.Tensor) -> torch.Tensor:
        """Compute total energy for vaccine sequence.

        Args:
            sequence: Vaccine sequence (batch, n_sites) or (n_sites,)

        Returns:
            Total energy (batch,) or scalar
        """
        if sequence.dim() == 1:
            sequence = sequence.unsqueeze(0)

        # Compute component energies
        e_escape = self._escape_energy(sequence)
        e_manufacturing = self._manufacturing_energy(sequence)
        e_autoimmune = self._autoimmune_energy(sequence)
        e_breadth = self._breadth_energy(sequence)

        # Weighted sum
        total = (
            self.config.escape_weight * e_escape
            + self.config.manufacturing_weight * e_manufacturing
            + self.config.autoimmune_weight * e_autoimmune
            + self.config.breadth_weight * e_breadth
        )

        return total.squeeze() if total.size(0) == 1 else total

    def _escape_energy(self, sequence: torch.Tensor) -> torch.Tensor:
        """Compute escape risk energy.

        Lower energy = lower escape probability.

        Args:
            sequence: Vaccine sequence (batch, n_sites)

        Returns:
            Escape energy (batch,)
        """
        sequence.size(0)

        if self.escape_predictor is not None:
            # Use learned predictor
            escape_prob = self.escape_predictor(sequence)
            return escape_prob.mean(dim=-1)

        # Fallback: estimate based on variant similarity
        # Vaccine closer to conserved positions = better
        one_hot = F.one_hot(sequence.long(), self.config.n_states).float()

        # Compare to variants
        variant_one_hot = F.one_hot(self.variant_profiles.long(), self.config.n_states).float()

        # Coverage: how many variants are "covered" by vaccine
        # Higher similarity to variants = better coverage
        n_variants = self.variant_profiles.size(0)

        coverage_scores = []
        for v in range(n_variants):
            # Position-wise match
            match = (one_hot * variant_one_hot[v]).sum(dim=-1)
            coverage_scores.append(match.mean(dim=-1))

        coverage = torch.stack(coverage_scores, dim=-1).mean(dim=-1)

        # Escape energy: inverse of coverage
        return 1 - coverage

    def _manufacturing_energy(self, sequence: torch.Tensor) -> torch.Tensor:
        """Compute manufacturing cost energy.

        Args:
            sequence: Vaccine sequence (batch, n_sites)

        Returns:
            Manufacturing energy (batch,)
        """
        # Sum of per-position manufacturing costs
        costs = self.manufacturing_costs[sequence.long()]
        return costs.mean(dim=-1)

    def _autoimmune_energy(self, sequence: torch.Tensor) -> torch.Tensor:
        """Compute autoimmune risk energy using Goldilocks zone.

        Vaccines should have intermediate self-similarity:
        - Too similar to self = autoimmune risk
        - Too dissimilar = poor efficacy

        Args:
            sequence: Vaccine sequence (batch, n_sites)

        Returns:
            Autoimmune energy (batch,)
        """
        sequence.size(0)

        # Compute self-similarity (placeholder: conservation-based)
        one_hot = F.one_hot(sequence.long(), self.config.n_states).float()

        # Weight by conservation (proxy for "selfness")
        self_similarity = (one_hot.mean(dim=-1) * self.conservation).mean(dim=-1)

        # Goldilocks score: optimal at center, penalty at extremes
        goldilocks_scores = compute_goldilocks_score(
            self_similarity,
            center=self.config.goldilocks_center,
            width=self.config.goldilocks_width,
        )

        # Energy: inverse of Goldilocks score
        return 1 - goldilocks_scores

    def _breadth_energy(self, sequence: torch.Tensor) -> torch.Tensor:
        """Compute breadth of protection energy.

        Lower energy = protection against more variants.

        Args:
            sequence: Vaccine sequence (batch, n_sites)

        Returns:
            Breadth energy (batch,)
        """
        sequence.size(0)
        n_variants = self.variant_profiles.size(0)

        # Count variants with high coverage
        one_hot = F.one_hot(sequence.long(), self.config.n_states).float()
        variant_one_hot = F.one_hot(self.variant_profiles.long(), self.config.n_states).float()

        variants_covered = []
        for v in range(n_variants):
            match = (one_hot * variant_one_hot[v]).sum(dim=-1).mean(dim=-1)
            variants_covered.append(match > 0.5)  # Threshold for "covered"

        breadth = torch.stack(variants_covered, dim=-1).float().mean(dim=-1)

        # Energy: inverse of breadth
        return 1 - breadth

    def compute_all_components(self, sequence: torch.Tensor) -> dict[ObjectiveType, torch.Tensor]:
        """Compute all energy components separately.

        Args:
            sequence: Vaccine sequence

        Returns:
            Dict mapping objective type to energy
        """
        if sequence.dim() == 1:
            sequence = sequence.unsqueeze(0)

        return {
            ObjectiveType.ESCAPE: self._escape_energy(sequence),
            ObjectiveType.MANUFACTURING: self._manufacturing_energy(sequence),
            ObjectiveType.AUTOIMMUNE: self._autoimmune_energy(sequence),
            ObjectiveType.BREADTH: self._breadth_energy(sequence),
        }


class VaccineOptimizer:
    """Vaccine optimizer using spin glass sampling.

    Uses replica exchange Monte Carlo to explore the vaccine
    fitness landscape and find optimal candidates.

    Example:
        >>> config = VaccineOptimizerConfig(n_sites=100)
        >>> optimizer = VaccineOptimizer(config)
        >>> result = optimizer.optimize(variant_profiles)
        >>> print(result.best_candidate.to_amino_acids())
    """

    def __init__(
        self,
        config: VaccineOptimizerConfig | None = None,
        escape_predictor: nn.Module | None = None,
    ):
        """Initialize vaccine optimizer.

        Args:
            config: Optimizer configuration
            escape_predictor: Optional escape probability predictor
        """
        self.config = config or VaccineOptimizerConfig()
        self.escape_predictor = escape_predictor

        # Replica exchange sampler
        self.sampler = ReplicaExchange(
            n_replicas=self.config.n_replicas,
            temp_min=self.config.temp_min,
            temp_max=self.config.temp_max,
            n_sweeps=10,
            exchange_frequency=self.config.exchange_frequency,
        )

    def optimize(
        self,
        variant_profiles: torch.Tensor | None = None,
        initial_sequence: torch.Tensor | None = None,
        constraints: dict | None = None,
    ) -> OptimizationResult:
        """Run vaccine optimization.

        Args:
            variant_profiles: Known variant sequences (n_variants, n_sites)
            initial_sequence: Starting sequence (n_sites,)
            constraints: Optional constraints (fixed positions, etc.)

        Returns:
            Optimization result with best candidates
        """
        # Create landscape
        landscape = ImmunogenicityLandscape(
            config=self.config,
            variant_profiles=variant_profiles,
            escape_predictor=self.escape_predictor,
        )

        # Initialize sequences for each replica
        if initial_sequence is not None:
            initial_configs = [initial_sequence.clone() for _ in range(self.config.n_replicas)]
            # Add noise to non-lowest temperature replicas
            for i in range(1, self.config.n_replicas):
                noise_fraction = 0.1 * i / self.config.n_replicas
                n_noisy = int(self.config.n_sites * noise_fraction)
                positions = torch.randperm(self.config.n_sites)[:n_noisy]
                initial_configs[i][positions] = torch.randint(self.config.n_states, (n_noisy,))
        else:
            initial_configs = None

        # Run optimization
        candidates, trajectory, stats = self._run_optimization(landscape, initial_configs, constraints)

        # Select best candidates
        candidates.sort(key=lambda c: c.energy)
        best = candidates[0]
        top_n = candidates[: self.config.n_candidates]

        # Find Pareto front
        pareto_front = self._compute_pareto_front(candidates)

        return OptimizationResult(
            best_candidate=best,
            all_candidates=top_n,
            optimization_trajectory=trajectory,
            acceptance_rates=stats["acceptance_rates"],
            exchange_rates=stats["exchange_rates"],
            pareto_front=pareto_front,
        )

    def _run_optimization(
        self,
        landscape: ImmunogenicityLandscape,
        initial_configs: list[torch.Tensor] | None,
        constraints: dict | None,
    ) -> tuple[list[VaccineCandidate], list[float], dict]:
        """Run the optimization loop.

        Args:
            landscape: Energy landscape
            initial_configs: Initial configurations
            constraints: Optimization constraints

        Returns:
            Tuple of (candidates, trajectory, statistics)
        """
        n_sites = self.config.n_sites
        n_states = self.config.n_states

        # Initialize configurations
        if initial_configs is None:
            configs = [torch.randint(n_states, (n_sites,)) for _ in range(self.config.n_replicas)]
        else:
            configs = [c.clone() for c in initial_configs]

        # Apply constraints
        fixed_positions = []
        if constraints is not None:
            fixed_positions = constraints.get("fixed_positions", [])

        # Statistics tracking
        trajectory = []
        acceptance_counts = {t.item(): 0 for t in self.sampler.temperatures}
        total_moves = {t.item(): 0 for t in self.sampler.temperatures}
        exchange_rates = []

        # Collect candidates
        candidates = []
        candidate_hashes = set()

        # Main optimization loop
        for iteration in range(self.config.n_iterations):
            # Metropolis sweeps at each temperature
            for rep in range(self.config.n_replicas):
                temp = self.sampler.temperatures[rep].item()
                config, n_accepted = self._metropolis_sweep(landscape, configs[rep], temp, fixed_positions)
                configs[rep] = config
                acceptance_counts[temp] += n_accepted
                total_moves[temp] += n_sites

            # Replica exchanges
            if iteration % self.config.exchange_frequency == 0:
                n_exchanges = 0
                for i in range(self.config.n_replicas - 1):
                    if self._attempt_exchange(landscape, configs, i, i + 1):
                        n_exchanges += 1
                exchange_rates.append(n_exchanges / (self.config.n_replicas - 1))

            # Record best energy
            energies = [landscape.energy(c.unsqueeze(0)).item() for c in configs]
            trajectory.append(min(energies))

            # Collect candidate from lowest temperature
            best_config = configs[0]
            config_hash = tuple(best_config.tolist())

            if config_hash not in candidate_hashes:
                candidate_hashes.add(config_hash)

                energy = landscape.energy(best_config.unsqueeze(0)).item()
                components = landscape.compute_all_components(best_config)

                candidate = VaccineCandidate(
                    sequence=best_config.clone(),
                    energy=energy,
                    component_energies={k: v.item() for k, v in components.items()},
                    escape_risk=components[ObjectiveType.ESCAPE].item(),
                    autoimmune_score=1 - components[ObjectiveType.AUTOIMMUNE].item(),
                    breadth_score=1 - components[ObjectiveType.BREADTH].item(),
                    metadata={"iteration": iteration},
                )
                candidates.append(candidate)

        # Compute acceptance rates
        acceptance_rates = {t: acceptance_counts[t] / max(1, total_moves[t]) for t in acceptance_counts}

        stats = {
            "acceptance_rates": acceptance_rates,
            "exchange_rates": exchange_rates,
        }

        return candidates, trajectory, stats

    def _metropolis_sweep(
        self,
        landscape: ImmunogenicityLandscape,
        configuration: torch.Tensor,
        temperature: float,
        fixed_positions: list[int],
    ) -> tuple[torch.Tensor, int]:
        """Perform one Metropolis sweep.

        Args:
            landscape: Energy landscape
            configuration: Current configuration
            temperature: Temperature
            fixed_positions: Positions that cannot be changed

        Returns:
            New configuration, number of accepted moves
        """
        config = configuration.clone()
        n_accepted = 0

        for site in range(self.config.n_sites):
            if site in fixed_positions:
                continue

            old_state = config[site].item()
            current_energy = landscape.energy(config.unsqueeze(0)).item()

            # Propose new state
            new_state = torch.randint(self.config.n_states, (1,)).item()
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
        landscape: ImmunogenicityLandscape,
        configs: list[torch.Tensor],
        i: int,
        j: int,
    ) -> bool:
        """Attempt replica exchange between i and j.

        Args:
            landscape: Energy landscape
            configs: List of configurations
            i, j: Replica indices

        Returns:
            Whether exchange was accepted
        """
        Ti = self.sampler.temperatures[i].item()
        Tj = self.sampler.temperatures[j].item()
        Ei = landscape.energy(configs[i].unsqueeze(0)).item()
        Ej = landscape.energy(configs[j].unsqueeze(0)).item()

        # Exchange criterion
        delta = (1 / Ti - 1 / Tj) * (Ej - Ei)

        if delta > 0 or torch.rand(1).item() < math.exp(delta):
            configs[i], configs[j] = configs[j].clone(), configs[i].clone()
            return True
        return False

    def _compute_pareto_front(self, candidates: list[VaccineCandidate]) -> list[VaccineCandidate]:
        """Compute Pareto-optimal candidates.

        A candidate is Pareto-optimal if no other candidate
        dominates it on all objectives.

        Args:
            candidates: All candidates

        Returns:
            Pareto-optimal candidates
        """
        if not candidates:
            return []

        pareto = []

        for candidate in candidates:
            is_dominated = False

            for other in candidates:
                if other is candidate:
                    continue

                # Check if other dominates candidate
                dominates = True
                strictly_better = False

                for obj in ObjectiveType:
                    if obj not in candidate.component_energies:
                        continue

                    c_val = candidate.component_energies[obj]
                    o_val = other.component_energies.get(obj, float("inf"))

                    if o_val > c_val:
                        dominates = False
                        break
                    if o_val < c_val:
                        strictly_better = True

                if dominates and strictly_better:
                    is_dominated = True
                    break

            if not is_dominated:
                pareto.append(candidate)

        return pareto

    def optimize_with_parisi_analysis(
        self,
        variant_profiles: torch.Tensor | None = None,
        n_samples: int = 100,
    ) -> tuple[OptimizationResult, dict]:
        """Optimize with Parisi overlap analysis.

        Uses replica symmetry breaking analysis to identify
        stable vaccine "phases" in the fitness landscape.

        Args:
            variant_profiles: Known variants
            n_samples: Samples for Parisi analysis

        Returns:
            Tuple of (optimization result, Parisi analysis)
        """
        # Run standard optimization
        result = self.optimize(variant_profiles)

        # Sample multiple configurations for overlap analysis
        landscape = ImmunogenicityLandscape(
            config=self.config,
            variant_profiles=variant_profiles,
            escape_predictor=self.escape_predictor,
        )

        samples = []
        for _ in range(n_samples):
            init = torch.randint(self.config.n_states, (self.config.n_sites,))
            # Cool down from high temperature
            for temp in [10.0, 5.0, 2.0, 1.0, 0.5, 0.1]:
                for _ in range(50):
                    init, _ = self._metropolis_sweep(landscape, init, temp, [])
            samples.append(init)

        # Compute overlap distribution
        overlaps = []
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                overlap = (samples[i] == samples[j]).float().mean().item()
                overlaps.append(overlap)

        # Analyze overlap distribution
        import numpy as np

        overlaps = np.array(overlaps)

        parisi_analysis = {
            "mean_overlap": float(np.mean(overlaps)),
            "std_overlap": float(np.std(overlaps)),
            "n_peaks": len(np.unique(np.round(overlaps, 1))),
            "is_rsb": float(np.std(overlaps)) > 0.1,  # Replica symmetry breaking
            "overlap_histogram": np.histogram(overlaps, bins=20),
        }

        return result, parisi_analysis


__all__ = [
    "ObjectiveType",
    "VaccineCandidate",
    "VaccineOptimizerConfig",
    "OptimizationResult",
    "ImmunogenicityLandscape",
    "VaccineOptimizer",
]
