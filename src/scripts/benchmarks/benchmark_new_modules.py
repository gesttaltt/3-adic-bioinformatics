#!/usr/bin/env python3
# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Performance benchmarks for newly added modules.

Benchmarks:
- graphs: Hyperbolic GNN operations
- topology: Persistent homology computation
- equivariant: SO(3)/SE(3) layer forward passes
- diffusion: Noise scheduling and denoising
- meta: MAML adaptation steps
- physics: Spin glass energy computation
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable

import torch
import torch.nn as nn

# Benchmark configuration
WARMUP_RUNS = 3
BENCHMARK_RUNS = 10
BATCH_SIZE = 32
SEQ_LEN = 100
HIDDEN_DIM = 64


def time_function(fn: Callable, *args, **kwargs) -> tuple[float, float]:
    """Time a function with warmup.

    Returns:
        Tuple of (mean_time_ms, std_time_ms)
    """
    # Warmup
    for _ in range(WARMUP_RUNS):
        fn(*args, **kwargs)

    # Benchmark
    times = []
    for _ in range(BENCHMARK_RUNS):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5
    return mean_time, std_time


def benchmark_graphs():
    """Benchmark hyperbolic GNN operations."""
    print("\n=== Graphs Module (Hyperbolic GNN) ===")
    results = {}

    try:
        from src.graphs import HyperbolicLinear, LorentzOperations, PoincareOperations

        # Poincare operations
        poincare = PoincareOperations(curvature=1.0)
        x = torch.randn(BATCH_SIZE, HIDDEN_DIM) * 0.3
        y = torch.randn(BATCH_SIZE, HIDDEN_DIM) * 0.3

        mean, std = time_function(poincare.distance, x, y)
        results["poincare_distance"] = (mean, std)
        print(f"  Poincare distance: {mean:.3f} +/- {std:.3f} ms")

        mean, std = time_function(poincare.mobius_add, x, y)
        results["mobius_add"] = (mean, std)
        print(f"  Mobius addition: {mean:.3f} +/- {std:.3f} ms")

        # Lorentz operations
        lorentz = LorentzOperations(curvature=1.0)
        space = torch.randn(BATCH_SIZE, HIDDEN_DIM - 1) * 0.3
        time_coord = torch.sqrt(1 + (space**2).sum(dim=-1, keepdim=True))
        x_lorentz = torch.cat([time_coord, space], dim=-1)

        mean, std = time_function(lorentz.minkowski_inner, x_lorentz, x_lorentz)
        results["minkowski_inner"] = (mean, std)
        print(f"  Minkowski inner product: {mean:.3f} +/- {std:.3f} ms")

        # Hyperbolic linear layer
        layer = HyperbolicLinear(HIDDEN_DIM, HIDDEN_DIM, curvature=1.0)
        x = torch.randn(BATCH_SIZE, HIDDEN_DIM) * 0.3

        mean, std = time_function(layer, x)
        results["hyperbolic_linear"] = (mean, std)
        print(f"  Hyperbolic linear: {mean:.3f} +/- {std:.3f} ms")

    except ImportError as e:
        print(f"  Skipped: {e}")
    except Exception as e:
        print(f"  Error: {e}")

    return results


def benchmark_topology():
    """Benchmark persistent homology operations."""
    print("\n=== Topology Module (Persistent Homology) ===")
    results = {}

    try:
        from src.topology import PersistenceVectorizer, RipsFiltration

        # Random point cloud
        points = torch.randn(50, 3)  # 50 points in 3D

        # Rips filtration - uses 'build' method
        filtration = RipsFiltration(max_edge_length=2.0)

        def compute_diagram():
            return filtration.build(points)

        mean, std = time_function(compute_diagram)
        results["rips_filtration"] = (mean, std)
        print(f"  Rips filtration (50 points): {mean:.3f} +/- {std:.3f} ms")

        # Persistence vectorizer
        fingerprint = compute_diagram()
        vectorizer = PersistenceVectorizer(n_bins=50)

        mean, std = time_function(vectorizer.vectorize, fingerprint)
        results["persistence_vectorize"] = (mean, std)
        print(f"  Persistence vectorization: {mean:.3f} +/- {std:.3f} ms")

    except ImportError as e:
        print(f"  Skipped: {e}")
    except Exception as e:
        print(f"  Error: {e}")

    return results


def benchmark_equivariant():
    """Benchmark equivariant network operations."""
    print("\n=== Equivariant Module (SO3/SE3 Networks) ===")
    results = {}

    try:
        from src.equivariant import SO3Layer, SphericalHarmonics

        # Spherical harmonics
        sh = SphericalHarmonics(lmax=2)
        directions = torch.randn(BATCH_SIZE, 3)
        directions = directions / directions.norm(dim=-1, keepdim=True)

        mean, std = time_function(sh, directions)
        results["spherical_harmonics"] = (mean, std)
        print(f"  Spherical harmonics (lmax=2): {mean:.3f} +/- {std:.3f} ms")

        # SO3 Layer - returns single tensor
        layer = SO3Layer(in_features=16, out_features=32, lmax=2)
        positions = torch.randn(BATCH_SIZE, 3)
        features = torch.randn(BATCH_SIZE, 16)
        # Simple k-NN graph
        n_edges = BATCH_SIZE * 10
        edge_index = torch.stack(
            [
                torch.randint(0, BATCH_SIZE, (n_edges,)),
                torch.randint(0, BATCH_SIZE, (n_edges,)),
            ]
        )

        def so3_forward():
            return layer(features, positions, edge_index)

        mean, std = time_function(so3_forward)
        results["so3_layer"] = (mean, std)
        print(f"  SO3 Layer forward: {mean:.3f} +/- {std:.3f} ms")

    except ImportError as e:
        print(f"  Skipped: {e}")
    except Exception as e:
        print(f"  Error: {e}")

    return results


def benchmark_diffusion():
    """Benchmark diffusion model operations."""
    print("\n=== Diffusion Module (Discrete Diffusion) ===")
    results = {}

    try:
        from src.diffusion import CodonDiffusion, NoiseScheduler

        # Noise scheduler - uses float input
        scheduler = NoiseScheduler(n_steps=1000, schedule_type="cosine")
        x = torch.randn(BATCH_SIZE, SEQ_LEN)  # Float noise
        t = torch.randint(0, 1000, (BATCH_SIZE,))

        mean, std = time_function(scheduler.add_noise, x, t)
        results["add_noise"] = (mean, std)
        print(f"  Add noise: {mean:.3f} +/- {std:.3f} ms")

        # Codon diffusion (forward pass only)
        model = CodonDiffusion(
            n_steps=100,  # Fewer steps for benchmarking
            vocab_size=64,
            hidden_dim=128,
            n_layers=2,
        )
        model.eval()

        # Discrete codon indices
        x_codons = torch.randint(0, 64, (BATCH_SIZE, SEQ_LEN))
        t = torch.randint(0, 100, (BATCH_SIZE,))

        with torch.no_grad():

            def diffusion_forward():
                return model.forward(x_codons, t)

            mean, std = time_function(diffusion_forward)
        results["diffusion_forward"] = (mean, std)
        print(f"  Diffusion forward: {mean:.3f} +/- {std:.3f} ms")

    except ImportError as e:
        print(f"  Skipped: {e}")
    except Exception as e:
        print(f"  Error: {e}")

    return results


def benchmark_meta():
    """Benchmark meta-learning operations."""
    print("\n=== Meta Module (MAML) ===")
    results = {}

    try:
        from src.meta import MAML, Task

        # Simple model
        model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 5))

        maml = MAML(model, inner_lr=0.01, n_inner_steps=3, first_order=True)

        # Create task
        task = Task(
            support_x=torch.randn(5, 10),
            support_y=torch.randint(0, 5, (5,)),
            query_x=torch.randn(15, 10),
            query_y=torch.randint(0, 5, (15,)),
        )

        mean, std = time_function(maml.adapt, task.support_x, task.support_y)
        results["maml_adapt"] = (mean, std)
        print(f"  MAML adapt (3 steps): {mean:.3f} +/- {std:.3f} ms")

        mean, std = time_function(maml, task)
        results["maml_forward"] = (mean, std)
        print(f"  MAML forward: {mean:.3f} +/- {std:.3f} ms")

    except ImportError as e:
        print(f"  Skipped: {e}")
    except Exception as e:
        print(f"  Error: {e}")

    return results


def benchmark_physics():
    """Benchmark statistical physics operations."""
    print("\n=== Physics Module (Spin Glass) ===")
    results = {}

    try:
        from src.physics import SpinGlassLandscape

        # Create spin glass landscape - uses n_sites, not n_spins
        landscape = SpinGlassLandscape(n_sites=64, n_states=2)

        # Random spin configuration as indices (0 or 1)
        spins = torch.randint(0, 2, (BATCH_SIZE, 64))

        mean, std = time_function(landscape.compute_energy, spins)
        results["spin_glass_energy"] = (mean, std)
        print(f"  Spin glass energy: {mean:.3f} +/- {std:.3f} ms")

        mean, std = time_function(landscape.compute_overlap, spins, spins)
        results["spin_glass_overlap"] = (mean, std)
        print(f"  Spin glass overlap: {mean:.3f} +/- {std:.3f} ms")

    except ImportError as e:
        print(f"  Skipped: {e}")
    except Exception as e:
        print(f"  Error: {e}")

    return results


def benchmark_information():
    """Benchmark information geometry operations."""
    print("\n=== Information Module (Fisher Geometry) ===")
    results = {}

    try:
        from src.information import FisherInformationEstimator

        # Simple model
        model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 5))

        estimator = FisherInformationEstimator(model)

        # Create a simple data loader
        def data_iter():
            for _ in range(10):
                yield (torch.randn(BATCH_SIZE, 10), torch.randint(0, 5, (BATCH_SIZE,)))

        def estimate_fisher():
            return estimator.estimate(data_iter(), n_samples=5)

        mean, std = time_function(estimate_fisher)
        results["fisher_estimation"] = (mean, std)
        print(f"  Fisher estimation: {mean:.3f} +/- {std:.3f} ms")

    except ImportError as e:
        print(f"  Skipped: {e}")
    except Exception as e:
        print(f"  Error: {e}")

    return results


def benchmark_contrastive():
    """Benchmark contrastive learning operations."""
    print("\n=== Contrastive Module (P-adic Contrastive) ===")
    results = {}

    try:
        from src.contrastive import PAdicContrastiveLoss

        loss_fn = PAdicContrastiveLoss(temperature=0.1, prime=3)

        # Random embeddings and p-adic indices
        embeddings = torch.randn(BATCH_SIZE, HIDDEN_DIM)
        indices = torch.arange(BATCH_SIZE)  # Sequential indices for p-adic valuation

        mean, std = time_function(loss_fn, embeddings, indices)
        results["contrastive_loss"] = (mean, std)
        print(f"  Contrastive loss: {mean:.3f} +/- {std:.3f} ms")

    except ImportError as e:
        print(f"  Skipped: {e}")
    except Exception as e:
        print(f"  Error: {e}")

    return results


def benchmark_tropical():
    """Benchmark tropical geometry operations."""
    print("\n=== Tropical Module (Tropical Geometry) ===")
    results = {}

    try:
        from src.tropical import TropicalPolynomial, TropicalSemiring

        semiring = TropicalSemiring(operation="min")

        # Tropical operations
        x = torch.randn(BATCH_SIZE, HIDDEN_DIM)
        y = torch.randn(BATCH_SIZE, HIDDEN_DIM)

        mean, std = time_function(semiring.add, x, y)
        results["tropical_add"] = (mean, std)
        print(f"  Tropical addition: {mean:.3f} +/- {std:.3f} ms")

        mean, std = time_function(semiring.multiply, x, y)
        results["tropical_multiply"] = (mean, std)
        print(f"  Tropical multiplication: {mean:.3f} +/- {std:.3f} ms")

    except ImportError as e:
        print(f"  Skipped: {e}")
    except Exception as e:
        print(f"  Error: {e}")

    return results


def benchmark_categorical():
    """Benchmark category theory operations."""
    print("\n=== Categorical Module (Category Theory) ===")
    results = {}

    try:
        from src.categorical import CategoricalLayer, TensorType

        # Create categorical layer
        in_type = TensorType(shape=(HIDDEN_DIM,), name="input")
        out_type = TensorType(shape=(HIDDEN_DIM,), name="output")
        layer = CategoricalLayer(in_type, out_type)

        x = torch.randn(BATCH_SIZE, HIDDEN_DIM)

        mean, std = time_function(layer, x)
        results["categorical_layer"] = (mean, std)
        print(f"  Categorical layer: {mean:.3f} +/- {std:.3f} ms")

    except ImportError as e:
        print(f"  Skipped: {e}")
    except Exception as e:
        print(f"  Error: {e}")

    return results


def print_summary(all_results: dict[str, dict[str, tuple[float, float]]]):
    """Print benchmark summary."""
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    all_ops = []
    for module, results in all_results.items():
        for op, (mean, std) in results.items():
            all_ops.append((module, op, mean, std))

    # Sort by time
    all_ops.sort(key=lambda x: x[2])

    print(f"\n{'Operation':<40} {'Time (ms)':<15} {'Module':<15}")
    print("-" * 70)
    for module, op, mean, std in all_ops:
        print(f"{op:<40} {mean:>6.3f} +/- {std:>5.3f}  {module:<15}")

    print("\n" + "=" * 60)
    print(f"Total operations benchmarked: {len(all_ops)}")
    print(f"Configuration: batch_size={BATCH_SIZE}, seq_len={SEQ_LEN}, hidden_dim={HIDDEN_DIM}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)


def main():
    """Run all benchmarks."""
    print("=" * 60)
    print("PERFORMANCE BENCHMARKS FOR NEW MODULES")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Sequence length: {SEQ_LEN}")
    print(f"  Hidden dimension: {HIDDEN_DIM}")
    print(f"  Warmup runs: {WARMUP_RUNS}")
    print(f"  Benchmark runs: {BENCHMARK_RUNS}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    all_results = {}

    all_results["graphs"] = benchmark_graphs()
    all_results["topology"] = benchmark_topology()
    all_results["equivariant"] = benchmark_equivariant()
    all_results["diffusion"] = benchmark_diffusion()
    all_results["meta"] = benchmark_meta()
    all_results["physics"] = benchmark_physics()
    all_results["information"] = benchmark_information()
    all_results["contrastive"] = benchmark_contrastive()
    all_results["tropical"] = benchmark_tropical()
    all_results["categorical"] = benchmark_categorical()

    print_summary(all_results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
