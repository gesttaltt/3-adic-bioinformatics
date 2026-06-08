#!/usr/bin/env python3
# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""B1: Pathogen-Specific AMP Design (Sequence-Space Optimization)

Research Idea Implementation - Carlos Brizuela

This module implements NSGA-II optimization in sequence space for designing
antimicrobial peptides targeting specific WHO priority pathogens. Each pathogen
has distinct membrane composition and optimal AMP characteristics.

UPDATED 2026-01-05: Now uses validated sequence-space mutations instead of
latent-space decoding. All candidates are real peptide sequences (10-35 AA).

Target Pathogens (WHO Priority):
1. Acinetobacter baumannii (Critical - Carbapenem-resistant)
2. Pseudomonas aeruginosa (Critical - MDR)
3. Enterobacteriaceae (Critical - Carbapenem-resistant)
4. Staphylococcus aureus (High - MRSA)
5. Helicobacter pylori (High - Clarithromycin-resistant)

Key Features:
1. Sequence-space NSGA-II (real peptide mutations)
2. PeptideVAE MIC prediction (Spearman r=0.74)
3. Pathogen-specific activity scoring
4. Confidence scoring based on hyperbolic radius
5. Multi-objective optimization (activity, toxicity, stability)

Usage:
    python scripts/B1_pathogen_specific_design.py --pathogen "S_aureus"
    python scripts/B1_pathogen_specific_design.py --all-pathogens --generations 100
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add paths - PACKAGE_DIR must be first to shadow project-level scripts/
SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent.parent
# Insert in reverse priority order so PACKAGE_DIR ends up at position 0
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "deliverables"))
sys.path.insert(0, str(PACKAGE_DIR))  # Must be first for scripts.* imports

# CRITICAL: Import scripts.* BEFORE shared - shared's import chain clobbers 'scripts' module
from scripts.sequence_nsga2 import (
    SequenceNSGA2,
    MutationOperators,
    ObjectiveFunctions,
    Peptide,
    OptimizationResult,
    MIN_PEPTIDE_LENGTH,
    MAX_PEPTIDE_LENGTH,
)

# Import from local src (self-contained)
from src.constants import AMINO_ACIDS, CHARGES, HYDROPHOBICITY

# Try to import predictor
try:
    from scripts.predict_mic import PeptideMICPredictor
    HAS_PREDICTOR = True
except ImportError:
    HAS_PREDICTOR = False


# =============================================================================
# Pathogen Definitions
# =============================================================================

WHO_PRIORITY_PATHOGENS = {
    "A_baumannii": {
        "full_name": "Acinetobacter baumannii",
        "gram": "negative",
        "priority": "critical",
        "resistance": "Carbapenem-resistant",
        "membrane_features": {
            "LPS_abundance": 0.85,
            "phosphatidylethanolamine": 0.70,
            "phosphatidylglycerol": 0.20,
            "cardiolipin": 0.05,
            "net_charge": -0.6,
        },
        "optimal_amp_features": {
            "net_charge": (4, 8),
            "hydrophobicity": (0.3, 0.5),
            "cationic_ratio": (0.25, 0.40),
            "length": (15, 30),
        },
        "seed_sequences": [
            "GIGKFLHSAKKFGKAFVGEIMNS",  # Magainin 2 (broad-spectrum)
            "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK",  # Cecropin A
            "RLARIVVIRVAR",  # LL-37 derivative
        ],
    },
    "P_aeruginosa": {
        "full_name": "Pseudomonas aeruginosa",
        "gram": "negative",
        "priority": "critical",
        "resistance": "MDR",
        "membrane_features": {
            "LPS_abundance": 0.90,
            "phosphatidylethanolamine": 0.65,
            "phosphatidylglycerol": 0.25,
            "cardiolipin": 0.08,
            "net_charge": -0.7,
        },
        "optimal_amp_features": {
            "net_charge": (5, 9),
            "hydrophobicity": (0.35, 0.55),
            "cationic_ratio": (0.30, 0.45),
            "length": (18, 35),
        },
        "seed_sequences": [
            "ILPWKWPWWPWRR",  # Indolicidin variant
            "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK",  # Cecropin A
            "RLKKTFFKIVKTVKW",  # BP100 derivative
        ],
    },
    "Enterobacteriaceae": {
        "full_name": "Enterobacteriaceae (E. coli, Klebsiella)",
        "gram": "negative",
        "priority": "critical",
        "resistance": "Carbapenem-resistant",
        "membrane_features": {
            "LPS_abundance": 0.88,
            "phosphatidylethanolamine": 0.72,
            "phosphatidylglycerol": 0.18,
            "cardiolipin": 0.06,
            "net_charge": -0.55,
        },
        "optimal_amp_features": {
            "net_charge": (3, 7),
            "hydrophobicity": (0.25, 0.45),
            "cationic_ratio": (0.20, 0.35),
            "length": (12, 25),
        },
        "seed_sequences": [
            "KLAKLAKKLAKLAK",  # LL-37 derivative
            "GIGKFLHSAKKFGKAFVGEIMNS",  # Magainin 2
            "KRFRIRVRV",  # Short cationic
        ],
    },
    "S_aureus": {
        "full_name": "Staphylococcus aureus (MRSA)",
        "gram": "positive",
        "priority": "high",
        "resistance": "Methicillin-resistant",
        "membrane_features": {
            "teichoic_acid": 0.40,
            "phosphatidylglycerol": 0.55,
            "lysyl_PG": 0.30,
            "cardiolipin": 0.10,
            "net_charge": -0.3,
        },
        "optimal_amp_features": {
            "net_charge": (2, 6),
            "hydrophobicity": (0.4, 0.6),
            "cationic_ratio": (0.20, 0.35),
            "length": (10, 22),
        },
        "seed_sequences": [
            "KLAKLAKKLAKLAK",  # LL-37 derivative
            "KLWKKLKKALK",  # Magainin derivative
            "FKCRRWQWRMKKLGAPS",  # Protegrin derivative
        ],
    },
    "H_pylori": {
        "full_name": "Helicobacter pylori",
        "gram": "negative",
        "priority": "high",
        "resistance": "Clarithromycin-resistant",
        "membrane_features": {
            "LPS_abundance": 0.75,
            "phosphatidylethanolamine": 0.60,
            "phosphatidylglycerol": 0.30,
            "cholesterol_glucosides": 0.15,
            "net_charge": -0.4,
        },
        "optimal_amp_features": {
            "net_charge": (2, 5),
            "hydrophobicity": (0.35, 0.50),
            "cationic_ratio": (0.18, 0.30),
            "length": (12, 20),
        },
        "seed_sequences": [
            "KLWKKLKKALK",  # Magainin derivative
            "KLAKLAKKLAKLAK",  # LL-37 derivative
            "RLKKTFFKIV",  # Short cationic
        ],
    },
}


# =============================================================================
# Pathogen-Specific Objectives
# =============================================================================

@dataclass
class PathogenCandidate:
    """AMP candidate with pathogen-specific scoring."""

    sequence: str
    mic_pred: float
    pathogen_score: float
    toxicity_pred: float
    stability_score: float
    confidence: str
    properties: Dict
    generation: int = 0

    def to_dict(self) -> Dict:
        return {
            "sequence": self.sequence,
            "length": len(self.sequence),
            "mic_pred": round(self.mic_pred, 4),
            "mic_ug_ml": round(10 ** self.mic_pred, 4),
            "mic_method": "ML-VALIDATED (PeptideVAE, Pearson r=0.61, p<0.001)",
            "pathogen_score": round(self.pathogen_score, 4),
            "pathogen_score_method": "HEURISTIC (membrane composition rules — not ML-validated)",
            "toxicity_pred": round(self.toxicity_pred, 4),
            "toxicity_method": "HEURISTIC (charge/hydrophobicity rules — not ML-validated)",
            "stability_score": round(self.stability_score, 4),
            "stability_method": "HEURISTIC (physicochemical rules — not ML-validated)",
            "confidence": self.confidence,
            "net_charge": round(self.properties.get("net_charge", 0), 2),
            "hydrophobicity": round(self.properties.get("hydrophobicity", 0), 3),
            "cationic_ratio": round(self.properties.get("cationic_ratio", 0), 3),
        }


class PathogenSpecificObjectives(ObjectiveFunctions):
    """Extended objectives with pathogen-specific scoring."""

    def __init__(
        self,
        pathogen: str,
        predictor: Optional["PeptideMICPredictor"] = None,
    ):
        super().__init__(predictor=predictor, use_heuristics=True)
        self.pathogen = pathogen
        self.pathogen_info = WHO_PRIORITY_PATHOGENS[pathogen]
        self.optimal = self.pathogen_info["optimal_amp_features"]
        self.membrane = self.pathogen_info["membrane_features"]

    def evaluate(self, sequence: str) -> Tuple[float, float, float, float]:
        """Evaluate all objectives for a sequence.

        Returns:
            Tuple of (mic_pred, pathogen_score, toxicity_pred, stability_score)
        """
        mic_pred, toxicity_pred, stability_score = super().evaluate(sequence)
        pathogen_score = self._pathogen_specific_score(sequence)

        return mic_pred, pathogen_score, toxicity_pred, stability_score

    def _pathogen_specific_score(self, sequence: str) -> float:
        """Calculate pathogen-specific activity score.

        Lower = better (closer to optimal for this pathogen).
        """
        n = len(sequence)
        if n == 0:
            return 10.0

        score = 0.0

        # Net charge deviation
        charge = sum(CHARGES.get(aa, 0) for aa in sequence)
        charge_min, charge_max = self.optimal["net_charge"]
        if charge < charge_min:
            score += (charge_min - charge) ** 2 * 0.1
        elif charge > charge_max:
            score += (charge - charge_max) ** 2 * 0.1

        # Hydrophobicity deviation
        hydro = sum(HYDROPHOBICITY.get(aa, 0) for aa in sequence) / n
        hydro_min, hydro_max = self.optimal["hydrophobicity"]
        if hydro < hydro_min:
            score += (hydro_min - hydro) ** 2
        elif hydro > hydro_max:
            score += (hydro - hydro_max) ** 2

        # Cationic ratio deviation
        cationic = sum(1 for aa in sequence if aa in "KRH") / n
        cat_min, cat_max = self.optimal["cationic_ratio"]
        if cationic < cat_min:
            score += (cat_min - cationic) ** 2 * 2
        elif cationic > cat_max:
            score += (cationic - cat_max) ** 2 * 2

        # Length deviation
        len_min, len_max = self.optimal["length"]
        if n < len_min:
            score += (len_min - n) * 0.05
        elif n > len_max:
            score += (n - len_max) * 0.03

        # Gram-specific bonuses
        if self.pathogen_info["gram"] == "negative":
            # Cationic peptides bind LPS better
            lps = self.membrane.get("LPS_abundance", 0.8)
            if charge >= 4:
                score -= 0.2 * lps
        else:
            # Gram-positive: teichoic acid interaction
            teichoic = self.membrane.get("teichoic_acid", 0.3)
            if hydro > 0.4:
                score -= 0.15 * teichoic

        return max(0, score)


# =============================================================================
# Pathogen-Specific Optimizer
# =============================================================================

class PathogenNSGA2(SequenceNSGA2):
    """NSGA-II optimizer specialized for pathogen-specific design."""

    def __init__(
        self,
        pathogen: str,
        population_size: int = 100,
        generations: int = 50,
        checkpoint_path: Optional[Path] = None,
        verbose: bool = True,
        random_seed: Optional[int] = None,
    ):
        if pathogen not in WHO_PRIORITY_PATHOGENS:
            raise ValueError(f"Unknown pathogen: {pathogen}. "
                           f"Available: {list(WHO_PRIORITY_PATHOGENS.keys())}")

        self.pathogen = pathogen
        self.pathogen_info = WHO_PRIORITY_PATHOGENS[pathogen]

        # Use pathogen-specific seeds
        seed_sequences = self.pathogen_info.get("seed_sequences", [
            "KLAKLAKKLAKLAK",
            "KLWKKLKKALK",
        ])

        super().__init__(
            seed_sequences=seed_sequences,
            population_size=population_size,
            generations=generations,
            checkpoint_path=checkpoint_path,
            verbose=verbose,
            random_seed=random_seed,
        )

        # Override objectives with pathogen-specific version
        self.objectives = PathogenSpecificObjectives(
            pathogen=pathogen,
            predictor=self.predictor,
        )

    def _setup_deap(self) -> None:
        """Configure DEAP for 4-objective optimization."""
        from scripts.sequence_nsga2 import deap, _lazy_import
        _lazy_import()

        # 4 objectives: minimize MIC, minimize pathogen_score, minimize toxicity, maximize stability
        if not hasattr(deap.creator, "FitnessPathogen"):
            deap.creator.create("FitnessPathogen", deap.base.Fitness, weights=(-1.0, -1.0, -1.0, 1.0))
        if not hasattr(deap.creator, "IndividualPathogen"):
            deap.creator.create("IndividualPathogen", list, fitness=deap.creator.FitnessPathogen)

        self.toolbox = deap.base.Toolbox()
        self.toolbox.register("individual", self._create_individual)
        self.toolbox.register("population", deap.tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", self._crossover)
        self.toolbox.register("mutate", self._mutate)
        self.toolbox.register("select", deap.tools.selNSGA2)

    def _create_individual(self):
        """Create individual with pathogen-specific fitness."""
        from scripts.sequence_nsga2 import deap
        base_seq = random.choice(self.seed_sequences)
        if random.random() < 0.5:
            mutated, _ = self.mutation_ops.mutate(base_seq)
            base_seq = mutated
        return deap.creator.IndividualPathogen([base_seq])

    def _evaluate_individual(self, individual) -> Tuple[float, float, float, float]:
        """Evaluate with 4 objectives."""
        sequence = individual[0]
        return self.objectives.evaluate(sequence)

    def run(self) -> List[PathogenCandidate]:
        """Run optimization and return pathogen candidates."""
        from scripts.sequence_nsga2 import deap

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"PATHOGEN-SPECIFIC AMP DESIGN: {self.pathogen_info['full_name']}")
            print(f"{'='*60}")
            print(f"Priority: {self.pathogen_info['priority']}")
            print(f"Resistance: {self.pathogen_info['resistance']}")
            print(f"Gram type: {self.pathogen_info['gram']}")
            print(f"Population: {self.population_size}, Generations: {self.generations}")
            print()

        # Initialize population
        population = self.toolbox.population(n=self.population_size)

        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Main evolution loop
        for gen in range(self.generations):
            # Assign crowding distance before tournament selection
            # (required for selTournamentDCD)
            fronts = deap.tools.sortNondominated(population, len(population))
            for front in fronts:
                deap.tools.emo.assignCrowdingDist(front)

            offspring = deap.tools.selTournamentDCD(population, len(population))
            offspring = [self.toolbox.clone(ind) for ind in offspring]

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < self.mutation_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            population = self.toolbox.select(population + offspring, self.population_size)

            if self.verbose and (gen + 1) % 10 == 0:
                pareto = deap.tools.sortNondominated(population, len(population), first_front_only=True)[0]
                best_mic = min(ind.fitness.values[0] for ind in pareto)
                print(f"Generation {gen+1}/{self.generations} | "
                      f"Pareto size: {len(pareto)} | Best MIC: {best_mic:.2f}")

        # Extract final Pareto front
        final_pareto = deap.tools.sortNondominated(population, len(population), first_front_only=True)[0]

        # Convert to PathogenCandidate objects
        candidates = []
        for ind in final_pareto:
            seq = ind[0]
            n = len(seq)
            props = {
                "net_charge": sum(CHARGES.get(aa, 0) for aa in seq),
                "hydrophobicity": sum(HYDROPHOBICITY.get(aa, 0) for aa in seq) / n if n > 0 else 0,
                "cationic_ratio": sum(1 for aa in seq if aa in "KRH") / n if n > 0 else 0,
            }

            # Determine confidence
            if self.predictor:
                try:
                    result = self.predictor.predict(seq)
                    confidence = result.confidence
                except Exception:
                    confidence = "Unknown"
            else:
                confidence = "Heuristic"

            candidates.append(PathogenCandidate(
                sequence=seq,
                mic_pred=ind.fitness.values[0],
                pathogen_score=ind.fitness.values[1],
                toxicity_pred=ind.fitness.values[2],
                stability_score=ind.fitness.values[3],
                confidence=confidence,
                properties=props,
                generation=self.generations,
            ))

        # Sort by MIC
        candidates.sort(key=lambda c: c.mic_pred)

        if self.verbose:
            print(f"\nOptimization complete: {len(candidates)} candidates for {self.pathogen}")

        return candidates


# =============================================================================
# Export Functions
# =============================================================================

def export_results(
    pathogen: str,
    candidates: List[PathogenCandidate],
    output_dir: Path,
) -> None:
    """Export optimization results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    pathogen_info = WHO_PRIORITY_PATHOGENS[pathogen]

    # JSON export
    results = {
        "pathogen": pathogen,
        "prediction_methods": {
            "mic_pred": "ML-VALIDATED (PeptideVAE, Pearson r=0.61, p<0.001)",
            "toxicity_pred": "HEURISTIC — NOT ML-VALIDATED (charge/hydrophobicity rules only)",
            "stability_score": "HEURISTIC — NOT ML-VALIDATED (physicochemical rules only)",
            "pathogen_score": "HEURISTIC — NOT ML-VALIDATED (membrane composition rules only)",
        },
        "pathogen_info": {
            "full_name": pathogen_info["full_name"],
            "gram": pathogen_info["gram"],
            "priority": pathogen_info["priority"],
            "resistance": pathogen_info["resistance"],
        },
        "n_candidates": len(candidates),
        "candidates": [c.to_dict() for c in candidates],
    }

    json_path = output_dir / f"{pathogen}_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Exported: {json_path}")

    # CSV export
    csv_path = output_dir / f"{pathogen}_candidates.csv"
    with open(csv_path, "w", newline="") as f:
        if candidates:
            writer = csv.DictWriter(f, fieldnames=list(candidates[0].to_dict().keys()))
            writer.writeheader()
            for c in candidates:
                writer.writerow(c.to_dict())
    print(f"Exported: {csv_path}")

    # FASTA export
    fasta_path = output_dir / f"{pathogen}_peptides.fasta"
    with open(fasta_path, "w") as f:
        for i, c in enumerate(candidates, 1):
            f.write(f">{pathogen}_rank{i:02d}_MIC{c.mic_pred:.2f}_{c.confidence}\n")
            f.write(f"{c.sequence}\n")
    print(f"Exported: {fasta_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pathogen-Specific AMP Design (Sequence-Space NSGA-II)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python B1_pathogen_specific_design.py --pathogen S_aureus
    python B1_pathogen_specific_design.py --all-pathogens --generations 100
    python B1_pathogen_specific_design.py --pathogen P_aeruginosa --output results/
        """,
    )

    parser.add_argument(
        "--pathogen",
        type=str,
        default="S_aureus",
        choices=list(WHO_PRIORITY_PATHOGENS.keys()),
        help="Target pathogen (default: S_aureus)",
    )
    parser.add_argument(
        "--all-pathogens",
        action="store_true",
        help="Run for all WHO priority pathogens",
    )
    parser.add_argument(
        "--generations", "-g",
        type=int,
        default=50,
        help="Number of generations (default: 50)",
    )
    parser.add_argument(
        "--population", "-p",
        type=int,
        default=100,
        help="Population size (default: 100)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=PACKAGE_DIR / "results" / "pathogen_specific",
        help="Output directory",
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=Path,
        help="Path to PeptideVAE checkpoint",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress messages",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("AMP DESIGN — PREDICTION METHOD NOTICE")
    print("=" * 60)
    print("  mic_pred       : ML-VALIDATED (PeptideVAE, r=0.61, p<0.001)")
    print("  toxicity_pred  : HEURISTIC [NOT ML-VALIDATED]")
    print("  stability_score: HEURISTIC [NOT ML-VALIDATED]")
    print("  pathogen_score : HEURISTIC [NOT ML-VALIDATED]")
    print("See BIAS_ANALYSIS.md for full methodology notes.")
    print("=" * 60)

    pathogens = list(WHO_PRIORITY_PATHOGENS.keys()) if args.all_pathogens else [args.pathogen]

    all_results = {}
    for pathogen in pathogens:
        optimizer = PathogenNSGA2(
            pathogen=pathogen,
            population_size=args.population,
            generations=args.generations,
            checkpoint_path=args.checkpoint,
            verbose=not args.quiet,
            random_seed=args.seed,
        )

        candidates = optimizer.run()
        export_results(pathogen, candidates, args.output)
        all_results[pathogen] = candidates

    # Summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    for pathogen, candidates in all_results.items():
        info = WHO_PRIORITY_PATHOGENS[pathogen]
        print(f"\n{info['full_name']}:")
        print(f"  Candidates: {len(candidates)}")
        if candidates:
            top = candidates[0]
            print(f"  Top: {top.sequence[:25]}{'...' if len(top.sequence) > 25 else ''}")
            print(f"  MIC: {10**top.mic_pred:.2f} ug/mL | Confidence: {top.confidence}")


if __name__ == "__main__":
    main()
