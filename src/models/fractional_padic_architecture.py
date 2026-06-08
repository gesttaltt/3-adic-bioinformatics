# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Fractional p-adic Architecture Design.

This module implements a continuous interpolation between p-adic bases,
allowing fractal dimensions like p=3.01, 3.05, 3.1, ..., eventually reaching
algebraic closure over both p=2 and p=3.

Mathematical Foundation:
------------------------
For integer p, we have:
- Operations: p^9
- Input dimension: p^2 (for 2D tables)
- Capacity requirement: 9 * log2(p) bits

For fractional p, we use continuous interpolation:
- Operations: floor(p^9) with soft boundaries
- Input dimension: interpolated between ceil(p^2) bases
- Capacity: 9 * log2(p) bits (continuous)

Key Insight:
-----------
The path from p=3 to p=6 (LCM of 2,3) passes through:
- p=3.0 (pure ternary)
- p=3.5 (halfway to quaternary)
- p=4.0 (quaternary)
- p=4.5
- p=5.0 (pentary)
- p=6.0 (closure over 2 AND 3)

At p=6, we have algebraic closure because:
- 6 = 2 × 3
- Any operation expressible in binary OR ternary is expressible in base-6
- This is the minimal complete p-adic system

Usage:
    python src/scripts/epsilon_vae/fractional_padic_architecture.py --p_values 3.0 3.1 3.5 4.0
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import OUTPUT_DIR


def compute_padic_dimensions(p: float) -> dict:
    """Compute architectural dimensions for fractional p.

    Args:
        p: The p-adic base (can be fractional)

    Returns:
        Dict with dimension specifications
    """
    # For ternary operations (truth tables), we have 3^3=27 inputs -> 3 outputs
    # Generalized: p^p inputs -> p outputs per "digit"
    # Full operation table: (p^2)^(p^2) but we use p^9 for 3-input functions

    # Number of operations (continuous extension)
    n_operations = p**9

    # Input dimension for encoding operations
    # For p=3: 9 (3x3 truth table flattened)
    # For fractional p: interpolate
    input_dim_exact = p**2
    input_dim = int(np.ceil(input_dim_exact))

    # Required information capacity (bits)
    required_bits = 9 * np.log2(p)

    # Recommended first layer width (to maintain full rank)
    # Need at least input_dim neurons to not lose information
    first_layer_width = max(input_dim, int(np.ceil(required_bits / np.log2(p))) * 16)

    # Latent dimension recommendation
    # Should be >= required_bits / log2(p) to encode all operations
    latent_dim = max(16, int(np.ceil(required_bits / 2)))  # 2 bits per dimension typical

    return {
        "p": p,
        "n_operations": n_operations,
        "n_operations_int": int(np.floor(n_operations)),
        "input_dim_exact": input_dim_exact,
        "input_dim": input_dim,
        "required_bits": required_bits,
        "first_layer_width": first_layer_width,
        "recommended_latent_dim": latent_dim,
        "is_integer_p": abs(p - round(p)) < 1e-10,
    }


def interpolate_operation_space(p: float, base_ops_3: np.ndarray) -> np.ndarray:
    """Interpolate operation space for fractional p.

    For p slightly above 3, we:
    1. Keep all 3^9 = 19683 ternary operations
    2. Add "interpolated" operations that represent fractional states
    3. The interpolation is based on the algebraic structure

    Args:
        p: Fractional p value (e.g., 3.1)
        base_ops_3: Base ternary operations (19683, 9)

    Returns:
        Extended operation array with interpolated operations
    """
    if p <= 3.0:
        return base_ops_3

    # Number of additional "fractional" operations
    n_base = len(base_ops_3)  # 19683
    n_target = int(np.floor(p**9))
    n_additional = n_target - n_base

    if n_additional <= 0:
        return base_ops_3

    # Create interpolated operations
    # These represent "fuzzy" states between discrete ternary values
    # Use convex combinations of existing operations

    rng = np.random.RandomState(42)  # Reproducible

    # Select pairs of operations to interpolate between
    idx1 = rng.randint(0, n_base, size=n_additional)
    idx2 = rng.randint(0, n_base, size=n_additional)

    # Interpolation weights (biased toward 0.5 for maximum novelty)
    alpha = rng.beta(2, 2, size=(n_additional, 1))

    # Interpolate
    interpolated = (1 - alpha) * base_ops_3[idx1] + alpha * base_ops_3[idx2]

    # Combine
    extended_ops = np.vstack([base_ops_3, interpolated])

    return extended_ops


class FractionalPadicEncoder(nn.Module):
    """Encoder that can handle fractional p-adic bases.

    Key design principles:
    1. Input layer scales with p^2
    2. Hidden layers maintain information capacity
    3. Latent dimension scales with log2(p^9) = 9*log2(p)
    """

    def __init__(
        self,
        p: float = 3.0,
        hidden_dim: int = 256,
        latent_dim: int = 16,
        base_input_dim: int = 9,  # For p=3
    ):
        super().__init__()

        self.p = p
        self.base_input_dim = base_input_dim

        # Compute dimensions for this p
        dims = compute_padic_dimensions(p)
        self.input_dim = dims["input_dim"]

        # Input projection (handles variable input size)
        # For p=3, this is identity-like
        # For p>3, this projects the extended input to standard size
        if self.input_dim != base_input_dim:
            self.input_proj = nn.Linear(self.input_dim, base_input_dim)
        else:
            self.input_proj = nn.Identity()

        # Scaling factor for capacity (continuous with p)
        capacity_scale = dims["required_bits"] / 14.3  # 14.3 is for p=3
        scaled_hidden = int(hidden_dim * capacity_scale)

        # Main encoder (same structure as TernaryVAE, but scaled)
        self.encoder = nn.Sequential(
            nn.Linear(base_input_dim, scaled_hidden),
            nn.LayerNorm(scaled_hidden),
            nn.GELU(),
            nn.Linear(scaled_hidden, scaled_hidden // 2),
            nn.LayerNorm(scaled_hidden // 2),
            nn.GELU(),
            nn.Linear(scaled_hidden // 2, scaled_hidden // 4),
            nn.LayerNorm(scaled_hidden // 4),
            nn.GELU(),
        )

        # Scaled latent dimension
        self.latent_dim = max(latent_dim, dims["recommended_latent_dim"])

        self.fc_mu = nn.Linear(scaled_hidden // 4, self.latent_dim)
        self.fc_logvar = nn.Linear(scaled_hidden // 4, self.latent_dim)

        # Store dimensions for reference
        self.dims = dims

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode operations to latent distribution.

        Args:
            x: Operations tensor, shape (batch, input_dim)

        Returns:
            mu, logvar of latent distribution
        """
        # Project input if needed
        x = self.input_proj(x)

        # Encode
        h = self.encoder(x)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar


def analyze_p_interpolation_path(
    p_values: list[float],
    output_dir: Path,
):
    """Analyze the interpolation path from p=3 to target.

    Args:
        p_values: List of p values to analyze
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    print(f"\n{'=' * 70}")
    print("FRACTIONAL P-ADIC DIMENSION ANALYSIS")
    print(f"{'=' * 70}")

    print(
        f"\n{'p':>6} | {'Operations':>12} | {'Input Dim':>10} | {'Bits Req':>10} | {'First Layer':>12} | {'Latent':>8}"
    )
    print("-" * 70)

    for p in p_values:
        dims = compute_padic_dimensions(p)
        results.append(dims)

        print(
            f"{p:>6.2f} | {dims['n_operations_int']:>12,} | {dims['input_dim']:>10} | "
            f"{dims['required_bits']:>10.1f} | {dims['first_layer_width']:>12} | {dims['recommended_latent_dim']:>8}"
        )

    # Key milestones
    print(f"\n{'=' * 70}")
    print("KEY MILESTONES")
    print(f"{'=' * 70}")

    milestones = {
        3.0: "Pure ternary (3-adic) - current system",
        4.0: "Quaternary (2^2-adic) - binary-compatible",
        6.0: "Senary (2×3-adic) - closure over binary AND ternary",
        math.sqrt(6): "Geometric mean of 2 and 3 - natural interpolation point",
    }

    for p, desc in sorted(milestones.items()):
        dims = compute_padic_dimensions(p)
        print(f"\n  p = {p:.4f}: {desc}")
        print(f"    Operations: {dims['n_operations_int']:,}")
        print(f"    Input dimension: {dims['input_dim']}")
        print(f"    Required bits: {dims['required_bits']:.1f}")

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    p_range = np.linspace(3.0, 6.0, 100)

    # Plot 1: Operations vs p
    ax1 = axes[0, 0]
    ops = [p**9 for p in p_range]
    ax1.semilogy(p_range, ops, "b-", linewidth=2)
    for p in [3, 4, 5, 6]:
        ax1.axvline(x=p, color="gray", linestyle="--", alpha=0.5)
        ax1.annotate(f"p={p}", (p, p**9), textcoords="offset points", xytext=(5, 5))
    ax1.set_xlabel("p")
    ax1.set_ylabel("Number of Operations (p^9)")
    ax1.set_title("Operation Space Size")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Required bits vs p
    ax2 = axes[0, 1]
    bits = [9 * np.log2(p) for p in p_range]
    ax2.plot(p_range, bits, "r-", linewidth=2)
    ax2.axhline(y=14.3, color="blue", linestyle="--", label="Current capacity (p=3)")
    ax2.fill_between(p_range, 14.3, bits, where=[b > 14.3 for b in bits], alpha=0.3, color="red", label="Capacity gap")
    ax2.set_xlabel("p")
    ax2.set_ylabel("Required Information Capacity (bits)")
    ax2.set_title("Information Requirements")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Input dimension vs p
    ax3 = axes[1, 0]
    input_dims = [int(np.ceil(p**2)) for p in p_range]
    ax3.plot(p_range, input_dims, "g-", linewidth=2)
    ax3.axhline(y=9, color="blue", linestyle="--", label="Current (p=3)")
    ax3.set_xlabel("p")
    ax3.set_ylabel("Input Dimension (ceil(p^2))")
    ax3.set_title("Input Layer Scaling")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Interpolation strategy
    ax4 = axes[1, 1]
    ax4.axis("off")

    strategy_text = """
FRACTIONAL P-ADIC INTERPOLATION STRATEGY

Phase 1: Optimize at p=3.0 (current)
  - Achieve 100% coverage with frozen encoder
  - Match v5_11_progressive performance
  - Establish baseline algebraic completeness

Phase 2: Continuous interpolation (p=3.0 → 3.5)
  - Increment p by 0.01-0.05 steps
  - Interpolate operation space
  - Scale encoder capacity proportionally
  - Monitor coverage at each step

Phase 3: Cross quaternary (p=3.5 → 4.0)
  - p=4 is binary-compatible (2^2)
  - Verify binary operations are representable
  - This is the first "closure checkpoint"

Phase 4: Approach senary (p=4.0 → 6.0)
  - p=6 = 2×3 gives closure over BOTH bases
  - At p=6, system can represent:
    * All binary (p=2) operations
    * All ternary (p=3) operations
    * All hexary (p=6) operations
  - This is FULL p-adic generalization

Key Property at p=6:
  Any integer n can be uniquely represented as:
    n = a₀ + a₁·6 + a₂·6² + ...
  where 0 ≤ aᵢ < 6

  Since 6 = 2×3, this subsumes both binary and ternary.
"""

    ax4.text(
        0.05, 0.95, strategy_text, transform=ax4.transAxes, fontsize=10, verticalalignment="top", fontfamily="monospace"
    )

    plt.tight_layout()
    plt.savefig(output_dir / "fractional_padic_analysis.png", dpi=150)
    plt.close()

    print(f"\nVisualization saved to {output_dir / 'fractional_padic_analysis.png'}")

    # Save results
    with open(output_dir / "fractional_padic_dimensions.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def create_interpolation_schedule(
    p_start: float = 3.0,
    p_end: float = 6.0,
    n_steps: int = 30,
    schedule_type: str = "logarithmic",
) -> list[float]:
    """Create a schedule for p-value interpolation.

    Args:
        p_start: Starting p value
        p_end: Ending p value
        n_steps: Number of interpolation steps
        schedule_type: "linear", "logarithmic", or "milestone"

    Returns:
        List of p values for training schedule
    """
    if schedule_type == "linear":
        return list(np.linspace(p_start, p_end, n_steps))

    elif schedule_type == "logarithmic":
        # More steps at lower p values (where changes are more sensitive)
        log_start = np.log(p_start)
        log_end = np.log(p_end)
        return list(np.exp(np.linspace(log_start, log_end, n_steps)))

    elif schedule_type == "milestone":
        # Focus on key algebraic milestones
        milestones = [
            3.0,  # Start (ternary)
            3.1,
            3.2,
            3.3,
            3.4,
            3.5,  # Fine steps to mid-point
            3.75,  # 3/4 toward 4
            4.0,  # Quaternary (binary-compatible)
            4.5,  # Mid toward pentary
            5.0,  # Pentary
            5.5,  # Mid toward senary
            6.0,  # Senary (full closure)
        ]
        return [m for m in milestones if p_start <= m <= p_end]

    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def main():
    parser = argparse.ArgumentParser(description="Fractional p-adic architecture analysis")
    parser.add_argument(
        "--p_values", nargs="+", type=float, default=[3.0, 3.1, 3.25, 3.5, 4.0, 5.0, 6.0], help="P values to analyze"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(OUTPUT_DIR / "epsilon_vae_analysis" / "fractional_padic"),
        help="Output directory",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Analyze interpolation path
    analyze_p_interpolation_path(args.p_values, output_dir)

    # Create interpolation schedule
    print(f"\n{'=' * 70}")
    print("RECOMMENDED INTERPOLATION SCHEDULE")
    print(f"{'=' * 70}")

    schedule = create_interpolation_schedule(3.0, 6.0, schedule_type="milestone")
    print("\nMilestone schedule (p=3 → p=6):")
    for i, p in enumerate(schedule):
        dims = compute_padic_dimensions(p)
        capacity_ratio = dims["required_bits"] / 14.3
        print(
            f"  Step {i + 1:2d}: p={p:.2f} → {dims['n_operations_int']:>10,} ops, capacity ratio: {capacity_ratio:.2f}x"
        )

    # Save schedule
    with open(output_dir / "interpolation_schedule.json", "w") as f:
        json.dump(
            {
                "schedule": schedule,
                "schedule_type": "milestone",
                "p_start": 3.0,
                "p_end": 6.0,
            },
            f,
            indent=2,
        )

    print(f"\n{'=' * 70}")
    print(f"Results saved to {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
