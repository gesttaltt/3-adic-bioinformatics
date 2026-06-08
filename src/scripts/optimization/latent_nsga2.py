# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""NSGA-II Multi-Objective Optimization in VAE Latent Space.

This module implements the NSGA-II algorithm operating directly on the latent
coordinates of a Ternary VAE, enabling smooth multi-objective optimization of
peptide properties without discrete sequence operations.

Key Objectives:
1. Maximize VAE Reconstruction Likelihood (Stability/Validity)
2. Minimize Toxicity Regressor Output (Safety)
3. Maximize Antimicrobial Activity (Efficacy)

Usage:
    python src/scripts/optimization/latent_nsga2.py \
        --vae_checkpoint models/ternary_vae.pt \
        --toxicity_model models/toxicity_regressor.pt \
        --output results/pareto_peptides.csv \
        --generations 100 \
        --population 200
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.paths import RESULTS_DIR

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


@dataclass
class Individual:
    """Represents an individual in the population."""

    latent: np.ndarray  # Latent vector (e.g., 16D)
    objectives: np.ndarray  # Objective values (to minimize)
    rank: int = 0  # Pareto rank
    crowding_distance: float = 0.0
    decoded_sequence: str | None = None


@dataclass
class OptimizationConfig:
    """Configuration for NSGA-II optimization."""

    latent_dim: int = 16
    population_size: int = 200
    generations: int = 100
    crossover_prob: float = 0.9
    mutation_prob: float = 0.1
    mutation_sigma: float = 0.1
    latent_bounds: tuple[float, float] = (-3.0, 3.0)
    seed: int = 42


class LatentNSGA2:
    """NSGA-II optimizer operating in VAE latent space.

    This optimizer performs multi-objective optimization directly on the
    latent coordinates of a VAE, allowing smooth exploration of the
    peptide design space.
    """

    def __init__(
        self,
        config: OptimizationConfig,
        objective_functions: list[Callable[[np.ndarray], float]],
        decoder: Callable[[np.ndarray], str] | None = None,
    ):
        """Initialize NSGA-II optimizer.

        Args:
            config: Optimization configuration
            objective_functions: List of objective functions (all to minimize)
            decoder: Optional function to decode latent to sequence
        """
        self.config = config
        self.objectives = objective_functions
        self.decoder = decoder
        self.n_objectives = len(objective_functions)

        np.random.seed(config.seed)

    def initialize_population(self) -> list[Individual]:
        """Create initial random population in latent space."""
        population = []
        for _ in range(self.config.population_size):
            latent = np.random.uniform(
                self.config.latent_bounds[0],
                self.config.latent_bounds[1],
                size=self.config.latent_dim,
            )
            individual = Individual(
                latent=latent,
                objectives=np.zeros(self.n_objectives),
            )
            population.append(individual)
        return population

    def evaluate_population(self, population: list[Individual]) -> None:
        """Evaluate objective functions for all individuals."""
        for ind in population:
            ind.objectives = np.array([obj(ind.latent) for obj in self.objectives])

    def fast_non_dominated_sort(self, population: list[Individual]) -> list[list[Individual]]:
        """Perform fast non-dominated sorting.

        Returns:
            List of fronts, where front[0] is the Pareto front
        """
        n = len(population)
        domination_count = [0] * n
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]

        for i, p in enumerate(population):
            for j, q in enumerate(population):
                if i == j:
                    continue
                if self._dominates(p, q):
                    dominated_solutions[i].append(j)
                elif self._dominates(q, p):
                    domination_count[i] += 1

            if domination_count[i] == 0:
                p.rank = 0
                fronts[0].append(p)

        current_front = 0
        while current_front < len(fronts) and fronts[current_front]:
            next_front = []
            for p_idx, p in enumerate(population):
                if p.rank == current_front:
                    for q_idx in dominated_solutions[p_idx]:
                        domination_count[q_idx] -= 1
                        if domination_count[q_idx] == 0:
                            population[q_idx].rank = current_front + 1
                            next_front.append(population[q_idx])
            current_front += 1
            if next_front:
                fronts.append(next_front)

        # Remove empty fronts
        return [f for f in fronts if f]

    def _dominates(self, p: Individual, q: Individual) -> bool:
        """Check if p dominates q (all objectives <= and at least one <)."""
        at_least_one_better = False
        for pi, qi in zip(p.objectives, q.objectives, strict=False):
            if pi > qi:
                return False
            if pi < qi:
                at_least_one_better = True
        return at_least_one_better

    def crowding_distance_assignment(self, front: list[Individual]) -> None:
        """Assign crowding distance to individuals in a front."""
        if len(front) <= 2:
            for ind in front:
                ind.crowding_distance = float("inf")
            return

        for ind in front:
            ind.crowding_distance = 0.0

        for m in range(self.n_objectives):
            front.sort(key=lambda x: x.objectives[m])

            # Boundary points get infinite distance
            front[0].crowding_distance = float("inf")
            front[-1].crowding_distance = float("inf")

            obj_range = front[-1].objectives[m] - front[0].objectives[m]
            if obj_range == 0:
                continue

            for i in range(1, len(front) - 1):
                front[i].crowding_distance += (front[i + 1].objectives[m] - front[i - 1].objectives[m]) / obj_range

    def select_parents(self, population: list[Individual]) -> tuple[Individual, Individual]:
        """Binary tournament selection based on rank and crowding distance."""

        def tournament(pop: list[Individual]) -> Individual:
            i, j = np.random.choice(len(pop), 2, replace=False)
            a, b = pop[i], pop[j]
            if a.rank < b.rank:
                return a
            elif b.rank < a.rank:
                return b
            else:
                return a if a.crowding_distance > b.crowding_distance else b

        return tournament(population), tournament(population)

    def crossover(self, parent1: Individual, parent2: Individual) -> tuple[Individual, Individual]:
        """Simulated Binary Crossover (SBX) for real-valued vectors."""
        if np.random.random() > self.config.crossover_prob:
            return (
                Individual(latent=parent1.latent.copy(), objectives=np.zeros(self.n_objectives)),
                Individual(latent=parent2.latent.copy(), objectives=np.zeros(self.n_objectives)),
            )

        eta = 20.0  # Distribution index
        child1_latent = np.zeros(self.config.latent_dim)
        child2_latent = np.zeros(self.config.latent_dim)

        for i in range(self.config.latent_dim):
            if np.random.random() < 0.5:
                if abs(parent1.latent[i] - parent2.latent[i]) > 1e-10:
                    if parent1.latent[i] < parent2.latent[i]:
                        y1, y2 = parent1.latent[i], parent2.latent[i]
                    else:
                        y1, y2 = parent2.latent[i], parent1.latent[i]

                    yl, yu = self.config.latent_bounds
                    rand = np.random.random()

                    beta = 1.0 + (2.0 * (y1 - yl) / (y2 - y1))
                    alpha = 2.0 - beta ** (-(eta + 1.0))
                    if rand <= 1.0 / alpha:
                        betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
                    else:
                        betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))

                    c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                    c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

                    child1_latent[i] = np.clip(c1, yl, yu)
                    child2_latent[i] = np.clip(c2, yl, yu)
                else:
                    child1_latent[i] = parent1.latent[i]
                    child2_latent[i] = parent2.latent[i]
            else:
                child1_latent[i] = parent1.latent[i]
                child2_latent[i] = parent2.latent[i]

        return (
            Individual(latent=child1_latent, objectives=np.zeros(self.n_objectives)),
            Individual(latent=child2_latent, objectives=np.zeros(self.n_objectives)),
        )

    def mutate(self, individual: Individual) -> Individual:
        """Polynomial mutation for real-valued vectors."""
        eta_m = 20.0  # Distribution index for mutation
        mutant = individual.latent.copy()

        for i in range(self.config.latent_dim):
            if np.random.random() < self.config.mutation_prob:
                y = mutant[i]
                yl, yu = self.config.latent_bounds
                delta = min(y - yl, yu - y) / (yu - yl)

                rand = np.random.random()
                if rand < 0.5:
                    deltaq = (2.0 * rand + (1.0 - 2.0 * rand) * (1.0 - delta) ** (eta_m + 1.0)) ** (
                        1.0 / (eta_m + 1.0)
                    ) - 1.0
                else:
                    deltaq = 1.0 - (2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (1.0 - delta) ** (eta_m + 1.0)) ** (
                        1.0 / (eta_m + 1.0)
                    )

                mutant[i] = np.clip(y + deltaq * (yu - yl), yl, yu)

        return Individual(latent=mutant, objectives=np.zeros(self.n_objectives))

    def run(self, verbose: bool = True) -> list[Individual]:
        """Run NSGA-II optimization.

        Returns:
            Pareto front individuals
        """
        # Initialize
        population = self.initialize_population()
        self.evaluate_population(population)

        for gen in range(self.config.generations):
            # Create offspring
            offspring = []
            while len(offspring) < self.config.population_size:
                p1, p2 = self.select_parents(population)
                c1, c2 = self.crossover(p1, p2)
                offspring.append(self.mutate(c1))
                if len(offspring) < self.config.population_size:
                    offspring.append(self.mutate(c2))

            self.evaluate_population(offspring)

            # Combine parent and offspring
            combined = population + offspring

            # Non-dominated sorting
            fronts = self.fast_non_dominated_sort(combined)

            # Select next generation
            population = []
            for front in fronts:
                self.crowding_distance_assignment(front)
                if len(population) + len(front) <= self.config.population_size:
                    population.extend(front)
                else:
                    # Sort by crowding distance and fill remaining
                    front.sort(key=lambda x: x.crowding_distance, reverse=True)
                    remaining = self.config.population_size - len(population)
                    population.extend(front[:remaining])
                    break

            if verbose and gen % 10 == 0:
                front_size = len([p for p in population if p.rank == 0])
                best_objs = np.min([p.objectives for p in population], axis=0)
                print(f"Gen {gen:4d}: Pareto front size = {front_size}, Best objectives = {best_objs}")

        # Decode final Pareto front if decoder available
        pareto_front = [p for p in population if p.rank == 0]
        if self.decoder:
            for ind in pareto_front:
                try:
                    ind.decoded_sequence = self.decoder(ind.latent)
                except Exception:
                    ind.decoded_sequence = None

        return pareto_front


def create_mock_objectives() -> list[Callable[[np.ndarray], float]]:
    """Create mock objective functions for testing."""

    def reconstruction_loss(z: np.ndarray) -> float:
        """Mock: Penalize extreme latent values."""
        return np.sum(z**2) / len(z)

    def toxicity(z: np.ndarray) -> float:
        """Mock: Simple toxicity based on latent structure."""
        return np.abs(np.mean(z[:8]) - np.mean(z[8:]))

    def activity(z: np.ndarray) -> float:
        """Mock: Negative activity (to minimize)."""
        return -np.std(z)  # Higher variance = higher activity

    return [reconstruction_loss, toxicity, activity]


def export_pareto_front(
    pareto_front: list[Individual],
    output_path: Path,
) -> None:
    """Export Pareto front to CSV."""
    if not HAS_PANDAS:
        print("pandas required for CSV export")
        return

    records = []
    for i, ind in enumerate(pareto_front):
        record = {
            "id": i,
            "rank": ind.rank,
            "crowding_distance": ind.crowding_distance,
        }

        # Add objectives
        for j, obj in enumerate(ind.objectives):
            record[f"objective_{j}"] = obj

        # Add latent dimensions
        for j, z in enumerate(ind.latent):
            record[f"z_{j}"] = z

        # Add decoded sequence if available
        if ind.decoded_sequence:
            record["sequence"] = ind.decoded_sequence

        records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"Exported {len(pareto_front)} Pareto-optimal solutions to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="NSGA-II Multi-Objective Optimization in VAE Latent Space")
    parser.add_argument(
        "--vae_checkpoint",
        type=str,
        default=None,
        help="Path to VAE checkpoint (optional, uses mock if not provided)",
    )
    parser.add_argument(
        "--toxicity_model",
        type=str,
        default=None,
        help="Path to toxicity regressor (optional)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(RESULTS_DIR / "pareto_peptides.csv"),
        help="Output CSV path for Pareto front",
    )
    parser.add_argument("--generations", type=int, default=100, help="Number of generations")
    parser.add_argument("--population", type=int, default=200, help="Population size")
    parser.add_argument("--latent_dim", type=int, default=16, help="Latent dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Configuration
    config = OptimizationConfig(
        latent_dim=args.latent_dim,
        population_size=args.population,
        generations=args.generations,
        seed=args.seed,
    )

    # Load models or use mocks
    if args.vae_checkpoint and HAS_TORCH and Path(args.vae_checkpoint).exists():
        print(f"Loading VAE from {args.vae_checkpoint}...")
        # TODO: Implement actual VAE loading
        objectives = create_mock_objectives()
        decoder = None
    else:
        print("Using mock objectives (no VAE checkpoint provided)")
        objectives = create_mock_objectives()
        decoder = None

    # Run optimization
    optimizer = LatentNSGA2(config, objectives, decoder)
    print("\nStarting NSGA-II optimization:")
    print(f"  Population: {config.population_size}")
    print(f"  Generations: {config.generations}")
    print(f"  Latent dim: {config.latent_dim}")
    print(f"  Objectives: {len(objectives)}")
    print()

    pareto_front = optimizer.run(verbose=True)

    # Export results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_pareto_front(pareto_front, output_path)

    print("\nOptimization complete!")
    print(f"  Pareto front size: {len(pareto_front)}")


if __name__ == "__main__":
    main()
