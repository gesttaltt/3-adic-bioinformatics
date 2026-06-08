# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Multi-objective optimization using Pareto dominance and NSGA-II.

This module implements evolutionary multi-objective optimization for
finding Pareto-optimal solutions across competing objectives.

Key Features:
- Fast non-dominated sorting (O(MN^2) where M=objectives, N=population)
- Crowding distance for diversity preservation
- NSGA-II selection and evolution operators
- Integration with ObjectiveRegistry

Usage:
    from src.training.optimizers import NSGAII, ParetoFrontOptimizer
    from src.objectives import ObjectiveRegistry

    nsga = NSGAII(population_size=100, n_generations=50)
    pareto_front = nsga.optimize(initial_population, objective_registry)

References:
    - Deb et al. (2002) "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II"
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass
class NSGAConfig:
    """Configuration for NSGA-II optimizer.

    Attributes:
        population_size: Number of individuals in population
        n_generations: Number of evolution generations
        crossover_rate: Probability of crossover
        mutation_rate: Probability of mutation
        mutation_scale: Scale of Gaussian mutation
        tournament_size: Size of tournament selection
    """

    population_size: int = 100
    n_generations: int = 50
    crossover_rate: float = 0.9
    mutation_rate: float = 0.1
    mutation_scale: float = 0.1
    tournament_size: int = 2


class ParetoFrontOptimizer:
    """Multi-objective optimizer using Pareto dominance.

    Used to select optimal latent candidates that balance multiple conflicting objectives:
    e.g., Minimize Reconstruction Loss vs Minimize Autoimmunity Risk.
    """

    def __init__(self):
        pass

    def is_dominated(self, candidate_scores: torch.Tensor, population_scores: torch.Tensor) -> bool:
        """Check if a candidate solution is dominated by any in the population.

        Args:
            candidate_scores: (N_Objectives,) tensor where LOWER is BETTER.
            population_scores: (Pop_Size, N_Objectives) tensor.

        Returns:
            True if dominated, False otherwise.
        """
        # A dominates B if A <= B for all objs AND A < B for at least one.
        # We assume minimization for all objectives.

        # Iterate efficiently?
        # Check if ANY individual in population dominates candidate

        # expand candidate to broadcast
        c = candidate_scores.unsqueeze(0)  # (1, N_obj)

        # domination_check: (Pop, N_obj) bools
        # better_or_equal: all(pop <= cand)
        better_or_equal = (population_scores <= c).all(dim=1)

        # strictly_better: any(pop < cand)
        strictly_better = (population_scores < c).any(dim=1)

        # dominated if there exists an individual that is better_or_equal AND strictly_better
        is_dominated = (better_or_equal & strictly_better).any()

        return bool(is_dominated.item())

    def identify_pareto_front(
        self, candidates: torch.Tensor, scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Identify the non-dominated set (Pareto Front) from a batch of candidates.

        Args:
            candidates: (Batch, Dim) - The actual latent vectors or solutions.
            scores: (Batch, N_Objectives) - The objective scores (LOWER is BETTER).

        Returns:
            (front_candidates, front_scores) containing only the non-dominated solutions.
        """
        batch_size = candidates.shape[0]
        is_dominated_mask = torch.zeros(batch_size, dtype=torch.bool, device=candidates.device)

        # Naive O(N^2) pairwise comparison
        # For large batches, NSGA-II fast non-dominated sorting is better, but this suffices for
        # typical inference batches (e.g. 512-2048).

        for i in range(batch_size):
            # Compare i against all others
            # We can vectorize the specific check for 'i' vs 'all others'

            c_scores = scores[i].unsqueeze(0)  # (1, Obj)

            # Others: excluding i? Or just include i (it won't dominate itself strictly)
            # Let's check against all

            better_or_equal = (scores <= c_scores).all(dim=1)
            strictly_better = (scores < c_scores).any(dim=1)
            dominators = better_or_equal & strictly_better

            if dominators.any():
                is_dominated_mask[i] = True

        # Filter
        non_dominated_indices = ~is_dominated_mask

        return candidates[non_dominated_indices], scores[non_dominated_indices]


def fast_non_dominated_sort(scores: torch.Tensor) -> list[torch.Tensor]:
    """Fast non-dominated sorting algorithm from NSGA-II.

    Assigns each individual to a Pareto front (rank).
    Front 0 = non-dominated, Front 1 = dominated only by Front 0, etc.

    Args:
        scores: Objective scores, shape (N, M) where N=population, M=objectives
                Lower is better.

    Returns:
        List of tensors, each containing indices of individuals in that front.
    """
    n = scores.shape[0]
    device = scores.device

    # Domination count and dominated set for each individual
    domination_count = torch.zeros(n, dtype=torch.long, device=device)
    dominated_by: list[list[int]] = [[] for _ in range(n)]

    # Compute domination relationships
    for i in range(n):
        for j in range(i + 1, n):
            # Check if i dominates j
            i_scores = scores[i]
            j_scores = scores[j]

            i_dom_j = (i_scores <= j_scores).all() and (i_scores < j_scores).any()
            j_dom_i = (j_scores <= i_scores).all() and (j_scores < i_scores).any()

            if i_dom_j:
                dominated_by[i].append(j)
                domination_count[j] += 1
            elif j_dom_i:
                dominated_by[j].append(i)
                domination_count[i] += 1

    # Build fronts
    fronts: list[torch.Tensor] = []
    remaining = set(range(n))

    while remaining:
        # Find individuals with zero domination count among remaining
        current_front = []
        for i in remaining:
            if domination_count[i] == 0:
                current_front.append(i)

        if not current_front:
            # All remaining are in a cycle - just take them all
            current_front = list(remaining)

        # Remove current front from remaining
        for i in current_front:
            remaining.discard(i)
            # Decrease domination count for individuals dominated by this one
            for j in dominated_by[i]:
                if j in remaining:
                    domination_count[j] -= 1

        fronts.append(torch.tensor(current_front, dtype=torch.long, device=device))

    return fronts


def compute_crowding_distance(scores: torch.Tensor) -> torch.Tensor:
    """Compute crowding distance for diversity preservation.

    Crowding distance measures how close an individual is to its neighbors
    in objective space. Higher distance = more isolated = more diverse.

    Args:
        scores: Objective scores for a single front, shape (N, M)

    Returns:
        Crowding distances, shape (N,)
    """
    n, m = scores.shape
    device = scores.device

    if n <= 2:
        # Too few individuals, all have infinite distance
        return torch.full((n,), float("inf"), device=device)

    distances = torch.zeros(n, device=device)

    for obj_idx in range(m):
        # Sort by this objective
        obj_scores = scores[:, obj_idx]
        sorted_indices = torch.argsort(obj_scores)

        # Boundary points get infinite distance
        distances[sorted_indices[0]] = float("inf")
        distances[sorted_indices[-1]] = float("inf")

        # Normalize by range
        obj_range = obj_scores[sorted_indices[-1]] - obj_scores[sorted_indices[0]]
        if obj_range < 1e-8:
            continue

        # Interior points
        for i in range(1, n - 1):
            idx = sorted_indices[i]
            prev_idx = sorted_indices[i - 1]
            next_idx = sorted_indices[i + 1]

            distance = (obj_scores[next_idx] - obj_scores[prev_idx]) / obj_range
            distances[idx] += distance

    return distances


class NSGAII:
    """NSGA-II Multi-objective Evolutionary Algorithm.

    Implements the Non-dominated Sorting Genetic Algorithm II for
    evolving latent space solutions across multiple objectives.

    Features:
    - Fast non-dominated sorting
    - Crowding distance for diversity
    - Binary tournament selection
    - SBX crossover and polynomial mutation
    """

    def __init__(self, config: NSGAConfig | None = None):
        """Initialize NSGA-II optimizer.

        Args:
            config: Configuration parameters (uses defaults if None)
        """
        self.config = config or NSGAConfig()
        self.pareto_optimizer = ParetoFrontOptimizer()

    def _tournament_select(
        self,
        population: torch.Tensor,
        scores: torch.Tensor,
        ranks: torch.Tensor,
        crowding: torch.Tensor,
    ) -> torch.Tensor:
        """Binary tournament selection based on rank and crowding distance.

        Args:
            population: Current population, shape (N, D)
            scores: Objective scores, shape (N, M)
            ranks: Front ranks, shape (N,)
            crowding: Crowding distances, shape (N,)

        Returns:
            Selected individual, shape (D,)
        """
        n = population.shape[0]
        k = self.config.tournament_size

        # Random tournament participants
        indices = torch.randperm(n)[:k]

        # Compare by rank first, then crowding distance
        best_idx = indices[0]
        for idx in indices[1:]:
            if ranks[idx] < ranks[best_idx] or ranks[idx] == ranks[best_idx] and crowding[idx] > crowding[best_idx]:
                best_idx = idx

        return population[best_idx]

    def _crossover(
        self,
        parent1: torch.Tensor,
        parent2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Simulated Binary Crossover (SBX).

        Args:
            parent1: First parent, shape (D,)
            parent2: Second parent, shape (D,)

        Returns:
            Two offspring tensors
        """
        if torch.rand(1).item() > self.config.crossover_rate:
            return parent1.clone(), parent2.clone()

        # SBX crossover
        eta = 20.0  # Distribution index
        u = torch.rand_like(parent1)

        beta = torch.where(
            u <= 0.5,
            (2 * u) ** (1 / (eta + 1)),
            (1 / (2 * (1 - u))) ** (1 / (eta + 1)),
        )

        child1 = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
        child2 = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)

        return child1, child2

    def _mutate(self, individual: torch.Tensor) -> torch.Tensor:
        """Polynomial mutation.

        Args:
            individual: Individual to mutate, shape (D,)

        Returns:
            Mutated individual
        """
        mutated = individual.clone()
        mask = torch.rand_like(individual) < self.config.mutation_rate

        # Gaussian mutation
        mutation = torch.randn_like(individual) * self.config.mutation_scale
        mutated = torch.where(mask, individual + mutation, individual)

        return mutated

    def _assign_ranks_and_crowding(
        self,
        population: torch.Tensor,
        scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assign front ranks and crowding distances to population.

        Args:
            population: Current population, shape (N, D)
            scores: Objective scores, shape (N, M)

        Returns:
            (ranks, crowding_distances) both shape (N,)
        """
        n = population.shape[0]
        device = population.device

        ranks = torch.zeros(n, dtype=torch.long, device=device)
        crowding = torch.zeros(n, device=device)

        fronts = fast_non_dominated_sort(scores)

        for rank, front_indices in enumerate(fronts):
            ranks[front_indices] = rank
            front_scores = scores[front_indices]
            front_crowding = compute_crowding_distance(front_scores)
            crowding[front_indices] = front_crowding

        return ranks, crowding

    def evolve_generation(
        self,
        population: torch.Tensor,
        scores: torch.Tensor,
        evaluate_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evolve one generation.

        Args:
            population: Current population, shape (N, D)
            scores: Current objective scores, shape (N, M)
            evaluate_fn: Function to evaluate new individuals

        Returns:
            (new_population, new_scores)
        """
        n = self.config.population_size

        # Assign ranks and crowding
        ranks, crowding = self._assign_ranks_and_crowding(population, scores)

        # Generate offspring
        offspring = []
        for _ in range(n // 2):
            # Select parents
            p1 = self._tournament_select(population, scores, ranks, crowding)
            p2 = self._tournament_select(population, scores, ranks, crowding)

            # Crossover
            c1, c2 = self._crossover(p1, p2)

            # Mutate
            c1 = self._mutate(c1)
            c2 = self._mutate(c2)

            offspring.extend([c1, c2])

        offspring_tensor = torch.stack(offspring[:n])

        # Evaluate offspring
        offspring_scores = evaluate_fn(offspring_tensor)

        # Combine parent and offspring
        combined = torch.cat([population, offspring_tensor], dim=0)
        combined_scores = torch.cat([scores, offspring_scores], dim=0)

        # Select next generation (elitist)
        combined_ranks, combined_crowding = self._assign_ranks_and_crowding(combined, combined_scores)

        # Sort by rank, then by crowding distance (descending)
        # Create sorting key: rank * 1e6 - crowding (to sort ascending by rank, descending by crowding)
        sort_key = combined_ranks.float() * 1e6 - combined_crowding
        sorted_indices = torch.argsort(sort_key)

        # Take top N
        selected = sorted_indices[:n]

        return combined[selected], combined_scores[selected]

    def optimize(
        self,
        initial_population: torch.Tensor,
        evaluate_fn: Callable[[torch.Tensor], torch.Tensor],
        callback: Callable[[int, torch.Tensor, torch.Tensor], None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run NSGA-II optimization.

        Args:
            initial_population: Starting population, shape (N, D)
            evaluate_fn: Function mapping population to scores, (N, D) -> (N, M)
            callback: Optional callback(generation, population, scores)

        Returns:
            (final_population, final_scores)
        """
        population = initial_population
        scores = evaluate_fn(population)

        for gen in range(self.config.n_generations):
            population, scores = self.evolve_generation(population, scores, evaluate_fn)

            if callback is not None:
                callback(gen, population, scores)

        return population, scores

    def get_pareto_front(
        self,
        population: torch.Tensor,
        scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract Pareto front from final population.

        Args:
            population: Final population, shape (N, D)
            scores: Final scores, shape (N, M)

        Returns:
            (front_population, front_scores) containing only non-dominated solutions
        """
        return self.pareto_optimizer.identify_pareto_front(population, scores)
