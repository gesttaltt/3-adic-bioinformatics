#!/usr/bin/env python3
"""Prove the -0.8321 hierarchy Spearman ceiling for 3^9 ternary operations.

CLAIM (from CLAUDE.md):
  The Spearman correlation between 3-adic valuations and hyperbolic radii cannot
  exceed -0.8321 with any within-level variance because v=0 contains 66.7% of
  samples (13,122 of 19,683).

PROOF STRATEGY:
  With tied valuation ranks (all v=k samples share the same rank) and freely
  assigned radius ranks, the maximum |Spearman rho| is:

      rho_max = -sqrt(SSBetween_val / SSTotal_rad) = -sqrt(eta_squared)

  where eta_squared is the correlation ratio (between-group SS / total SS)
  for the valuation variable.

  This bound is achieved when radius ranks are perfectly anti-correlated with
  valuation levels (highest v → smallest radius), with no radius rank overlap
  between levels.

DERIVATION:
  Let nk = number of samples at valuation v=k, n = sum(nk) = 3^9 = 19683.

  Valuation rank for group k (average of tied ranks):
      val_rank_k = sum of all ranks in [pos_start, pos_end] / nk
                 = (pos_start + pos_end) / 2

  For optimal negative correlation:
      rad_rank_k = n + 1 - val_rank_k

  This gives (rad_rank_k - mean) = -(val_rank_k - mean), so the numerator is:
      num = sum_k nk * (val_rank_k - mean) * (rad_rank_k - mean)
          = -sum_k nk * (val_rank_k - mean)^2
          = -SSBetween_val

  Denominator:
      sqrt(SSBetween_val * SSTotal_rad)

  where SSTotal_rad = n(n^2-1)/12 (standard formula for ranks 1..n).
  SSBetween_val = SSBetween_val (zero within-group variance for valuation, so
  total equals between).

  Therefore: rho = -SSBetween / sqrt(SSBetween * SSTotal) = -sqrt(SSBetween / SSTotal)

Usage:
    python src/scripts/analysis/hierarchy_ceiling_proof.py
"""

import numpy as np


def compute_3adic_valuation_distribution(p: int = 3, n_digits: int = 9) -> dict:
    """Compute the number of samples at each 3-adic valuation level.

    For x in {0, 1, ..., p^n - 1}:
      v_p(x) = k  iff  p^k | x  and  p^(k+1) does not divide x
      v_p(0) = n (by convention, the zero element has maximal valuation)

    Returns:
        dict mapping valuation -> count
    """
    total = p ** n_digits
    distribution = {}

    # v=k: divisible by p^k but not p^(k+1)
    for k in range(n_digits):
        count = p ** (n_digits - k) - p ** (n_digits - k - 1)
        distribution[k] = count

    # v=n_digits: only the element 0 itself
    distribution[n_digits] = 1

    assert sum(distribution.values()) == total, "Distribution must sum to total"
    return distribution


def compute_hierarchy_ceiling(p: int = 3, n_digits: int = 9) -> dict:
    """Compute the maximum achievable |Spearman rho| between valuations and radii.

    Returns a dict with the ceiling value and intermediate calculations.
    """
    dist = compute_3adic_valuation_distribution(p, n_digits)
    n = p ** n_digits
    mean_rank = (n + 1) / 2

    # Sort by valuation (ascending), which corresponds to largest radius first
    # for positive hierarchy, or smallest radius first for negative hierarchy.
    # For maximum NEGATIVE Spearman: assign smallest radii to highest valuations.
    levels = sorted(dist.keys())  # 0, 1, ..., n_digits

    # Compute valuation rank (average rank in sorted order)
    # Level 0 occupies the lowest positions (ranks 1..n0),
    # level 1 occupies ranks (n0+1)..(n0+n1), etc.
    val_rank = {}
    pos = 1
    for v in levels:
        nv = dist[v]
        val_rank[v] = pos + (nv - 1) / 2  # average rank
        pos += nv

    # For maximum negative Spearman: assign lowest radii to highest valuation.
    # Highest valuation = v=n_digits (rank position closest to n).
    # Flip: rad_rank_v = n + 1 - val_rank_v
    rad_rank = {v: n + 1 - val_rank[v] for v in levels}

    # SSBetween (valuation) = sum_v nv * (val_rank_v - mean)^2
    # This equals SSTotal_val because within-group variance is zero (all ties)
    ss_between_val = sum(dist[v] * (val_rank[v] - mean_rank) ** 2 for v in levels)

    # SSTotal for radius ranks (no ties assumed, ranks 1..n)
    # = n(n^2-1)/12
    ss_total_rad = n * (n ** 2 - 1) / 12

    # Verify that SSBetween_val <= SSTotal_val
    # SSTotal_val = SSBetween_val + SSWithin_val = SSBetween_val (since within = 0)
    ss_total_val = ss_between_val  # zero within-group variance

    # Maximum |Spearman rho|
    rho_ceiling = -np.sqrt(ss_between_val / ss_total_rad)

    # Cross-check: eta_squared = SSBetween / SSTotal_rad
    eta_squared = ss_between_val / ss_total_rad

    return {
        "p": p,
        "n_digits": n_digits,
        "n_total": n,
        "n_levels": len(levels),
        "distribution": {str(v): dist[v] for v in levels},
        "v0_fraction": dist[0] / n,
        "ss_between_val": ss_between_val,
        "ss_total_rad": ss_total_rad,
        "eta_squared": eta_squared,
        "rho_ceiling": rho_ceiling,
        "rho_ceiling_abs": abs(rho_ceiling),
    }


def print_proof(result: dict) -> None:
    print("=" * 70)
    print("HIERARCHY CEILING PROOF: 3-adic Valuation vs Hyperbolic Radius")
    print("=" * 70)

    p = result["p"]
    q = result["n_digits"]
    n = result["n_total"]
    dist = {int(k): v for k, v in result["distribution"].items()}

    print(f"\nSystem: p={p}, exponent={q}, total operations={n:,}")
    print(f"\nValuation distribution:")
    print(f"  {'v':>4}  {'count':>8}  {'fraction':>10}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*10}")
    for v in sorted(dist.keys()):
        frac = dist[v] / n
        print(f"  {v:>4}  {dist[v]:>8,}  {frac:>10.4f}")

    print(f"\nKey statistic: v=0 contains {result['v0_fraction']:.4f} "
          f"({dist[0]:,}/{n:,}) of all samples")

    print(f"\nSSBetween_val (valuation between-group SS) = {result['ss_between_val']:.0f}")
    print(f"SSTotal_rad   (radius total SS, ranks 1..n) = {result['ss_total_rad']:.0f}")
    print(f"eta^2 = SSBetween / SSTotal = {result['eta_squared']:.6f}")

    print(f"\nMaximum achievable Spearman correlation:")
    print(f"  rho_max = -sqrt(eta^2) = -sqrt({result['eta_squared']:.6f})")
    print(f"  rho_max = {result['rho_ceiling']:.6f}")
    print(f"  |rho_max| = {result['rho_ceiling_abs']:.6f}")

    print(f"\nDocumented ceiling in CLAUDE.md: -0.8321")
    print(f"Computed ceiling:               {result['rho_ceiling']:.4f}")

    tol = 0.001
    if abs(abs(result['rho_ceiling']) - 0.8321) < tol:
        print(f"\nVERIFIED: Computed ceiling matches documented value (within {tol})")
    else:
        print(f"\nDISCREPANCY: Computed {result['rho_ceiling']:.4f} vs documented -0.8321")

    print("\n" + "=" * 70)
    print("PROOF SKETCH")
    print("=" * 70)
    print("""
Given:
  - All members of valuation level v have the SAME valuation rank (tied).
  - Within-group variance of valuation is zero.
  - Radius ranks are freely assignable (no ties in the optimal case).

For maximum |rho| (negative correlation):
  - Assign smallest radii to highest valuations (v=9 -> rank 1, v=0 -> ranks 6562-19683).
  - With this assignment: (rad_rank_v - mean) = -(val_rank_v - mean) for each level v.

Numerator of Spearman:
  sum_i (val_rank_i - mean)(rad_rank_i - mean)
  = sum_v nv * (val_rank_v - mean) * (-(val_rank_v - mean))
  = -SSBetween_val

Denominator of Spearman:
  sqrt(SSTotal_val * SSTotal_rad)
  = sqrt(SSBetween_val * SSTotal_rad)      [SSTotal_val = SSBetween_val since within = 0]

Therefore:
  rho_max = -SSBetween_val / sqrt(SSBetween_val * SSTotal_rad)
           = -sqrt(SSBetween_val / SSTotal_rad)
           = -sqrt(eta^2)
           ≈ -0.8321

The ceiling is strictly less than -1 because SSTotal_rad includes both
between-group AND within-group variance in radii, while SSBetween_val
captures only between-group variance in valuations.
""")


def verify_with_simulation(n_samples: int = 1000) -> None:
    """Numerically verify the ceiling by constructing the optimal assignment."""
    p, q = 3, 9
    n = p ** q

    print("=" * 70)
    print("NUMERICAL VERIFICATION (optimal assignment simulation)")
    print("=" * 70)

    # Compute true valuations for all 3^9 = 19683 elements
    valuations = np.zeros(n, dtype=int)
    for x in range(1, n):
        v = 0
        tmp = x
        while tmp % p == 0:
            v += 1
            tmp //= p
        valuations[x] = v
    valuations[0] = q  # zero has maximal valuation

    # Optimal radius assignment: highest valuation -> smallest radius
    # Assign radius rank = n - argrank(valuation) + 1 within each group
    # (Simply: sort by valuation descending, assign radius ranks 1..n)
    order = np.argsort(-valuations, kind='stable')  # descending valuation
    radius_ranks = np.empty(n, dtype=float)
    for rank_pos, idx in enumerate(order):
        radius_ranks[idx] = rank_pos + 1

    from scipy.stats import spearmanr
    rho, p_val = spearmanr(valuations, radius_ranks)
    print(f"Spearman rho with optimal assignment: {rho:.6f}")
    print(f"Theoretical ceiling:                  {compute_hierarchy_ceiling()['rho_ceiling']:.6f}")

    diff = abs(rho - compute_hierarchy_ceiling()['rho_ceiling'])
    print(f"Difference:                           {diff:.2e}")
    if diff < 1e-4:
        print("MATCH: Simulation confirms the theoretical ceiling.")
    else:
        print("WARNING: Simulation and theory diverge — check implementation.")


if __name__ == "__main__":
    result = compute_hierarchy_ceiling(p=3, n_digits=9)
    print_proof(result)
    print()
    try:
        from scipy.stats import spearmanr
        verify_with_simulation()
    except ImportError:
        print("scipy not available — skipping numerical verification")
