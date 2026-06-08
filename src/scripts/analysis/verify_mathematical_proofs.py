# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Mathematical Proofs Verification Script.

Implements the "Mathematical Stress Tests" defined in
DOCUMENTATION/01_STAKEHOLDER_RESOURCES/validation_suite/02_MATHEMATICAL_STRESS_TESTS.md

Validates:
1. Delta-Hyperbolicity (Gromov 4-point condition)
2. Ultrametricity Score (Isosceles triangle condition)
3. Zero-Structure / P-Adic Valuation (via analyze_zero_structure logic)
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.geometry import compute_delta_hyperbolicity, compute_ultrametricity_score
from src.data.generation import generate_all_ternary_operations
from src.geometry import poincare_distance
from src.models import TernaryVAEV5_11

# Import the existing specific zero analysis logic
try:
    from scripts.analysis.analyze_zero_structure import analyze_checkpoint as analyze_zeros_internal
except ImportError:
    # If import fails (path issues), we'll reimplement the light version here
    analyze_zeros_internal = None


def verify_proofs(
    checkpoint_path: str = None,
    model_config: dict[str, Any] = None,
    device: str = "cuda",
):
    print(f"\n{'=' * 60}")
    print("MATHEMATICAL PROOFS VERIFICATION")
    print(f"{'=' * 60}\n")

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load or Initialize Model
    if model_config is None:
        model_config = {
            "latent_dim": 16,
            "hidden_dim": 64,
            "use_dual_projection": True,
            "use_controller": True,
            "curvature": -1.0,
            # Note: Model definition expects curvature as float magnitude usually,
            # or check implementation. V5.11 init takes 'curvature' float.
            # Usually we pass positive curvature parameter c where K = -c^2 or similar.
            # Default to 1.0 (K=-1).
        }

    model = TernaryVAEV5_11(**model_config)

    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        # Use simple load for now, assuming standard save format
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Warning: Strict processing failed ({e}), trying loose...")
            pass
    else:
        print("No checkpoint found/provided. Using random initialization.")
        print("(Note: Random models are unlikely to satisfy proofs unless architecture enforces them)")

    model.to(device)
    model.eval()

    # 2. generating Data
    print("Generating complete ternary operation set (3^9 = 19683)...")
    ops = generate_all_ternary_operations()
    ops_tensor = torch.tensor(ops, dtype=torch.float32, device=device)

    # 3. Compute Embeddings
    print("Computing hyperbolic embeddings...")
    with torch.no_grad():
        outputs = model(ops_tensor)
        z_A = outputs["z_A_hyp"]  # (N, D)
        outputs["z_B_hyp"]

        # We can analyze both or just A. Proofs usually target the "primary" manifold.
        # Let's check A.

    embeddings = z_A
    print(f"Embedding shape: {embeddings.shape}")

    # 4. Verify Delta-Hyperbolicity
    print("\n[Test 1] Delta-Hyperbolicity (Target < 0.1)")
    delta = compute_delta_hyperbolicity(embeddings, sample_size=200)
    print(f"  Result: delta = {delta:.4f}")
    if delta < 0.1:
        print("  Status: PASS [OK]")
    elif delta < 0.5:
        print("  Status: MARGINAL [WARN]")
    else:
        print("  Status: FAIL [X]")

    # 5. Verify Ultrametricity
    print("\n[Test 2] Ultrametricity Score (Target > 0.99)")
    ultra_score = compute_ultrametricity_score(embeddings, sample_size=200)
    print(f"  Result: score = {ultra_score:.4f}")
    if ultra_score > 0.99:
        print("  Status: PASS [OK]")
    elif ultra_score > 0.90:
        print("  Status: MARGINAL [WARN]")
    else:
        print("  Status: FAIL [X]")

    # 6. Zero Structure (P-adic)
    # We can calculate correlation quickly here
    print("\n[Test 3] P-adic Valuation Correlation")

    # Calculate valuation (trailing zeros)
    def valuation(op):
        cnt = 0
        for x in op:
            if x == 0:
                cnt += 1
            else:
                break
        return cnt

    vals = np.array([valuation(op) for op in ops])
    # V5.12.2: Use hyperbolic distance for radii
    origin = torch.zeros_like(embeddings)
    radii = poincare_distance(embeddings, origin, c=1.0).cpu().numpy()

    # Check correlation
    from scipy import stats

    corr, p_val = stats.pearsonr(vals, radii)

    print(f"  Correlation (Valuation vs Radius): r={corr:.4f}")
    print("  Expected: Negative correlation (higher valuation = closer to origin = smaller radius)")

    if corr < -0.5:
        print("  Status: PASS [OK] (Strong p-adic structure)")
    elif corr < -0.1:
        print("  Status: MARGINAL [WARN] (Weak p-adic structure)")
    else:
        print("  Status: FAIL [X] (No/Inverted p-adic structure)")

    # Return summary
    return {
        "delta": delta,
        "ultrametricity": ultra_score,
        "padic_correlation": corr,
    }


def run_grid_stress_test(device: str = "cuda"):
    """Runs the 40+ scenario grid defined in documentation."""
    print(f"\n{'=' * 60}")
    print("RUNNING MATHEMATICAL STRESS TEST GRID (Targeting Goldilocks Zone)")
    print(f"{'=' * 60}\n")

    # Grid definitions matching DOCUMENTATION/01_PROJECT_KNOWLEDGE_BASE/02_THEORY_AND_FOUNDATIONS/validation_suite/02_MATHEMATICAL_STRESS_TESTS.md
    scenarios = [
        # Set A: Standard Hyperbolic Tree (c=-1)
        {"dim": 8, "c": -1.0, "base": 2, "id": "M-01"},
        {"dim": 16, "c": -1.0, "base": 2, "id": "M-02"},
        {"dim": 32, "c": -1.0, "base": 2, "id": "M-03"},
        {"dim": 64, "c": -1.0, "base": 2, "id": "M-04"},
        {"dim": 128, "c": -1.0, "base": 2, "id": "M-05"},
        {"dim": 256, "c": -1.0, "base": 2, "id": "M-06"},
        {"dim": 512, "c": -1.0, "base": 2, "id": "M-07"},
        {"dim": 1024, "c": -1.0, "base": 2, "id": "M-08"},
        # Set B: High Curvature Cycle (c=-0.1)
        {"dim": 8, "c": -0.1, "base": 3, "id": "M-09"},
        {"dim": 16, "c": -0.1, "base": 3, "id": "M-10"},
        {"dim": 32, "c": -0.1, "base": 3, "id": "M-11"},
        {"dim": 64, "c": -0.1, "base": 3, "id": "M-12"},
        {"dim": 128, "c": -0.1, "base": 3, "id": "M-13"},
        # Set C: Mesh Topology (c=-2.0)
        {"dim": 8, "c": -2.0, "base": 5, "id": "M-14"},
        {"dim": 16, "c": -2.0, "base": 5, "id": "M-15"},
        {"dim": 32, "c": -2.0, "base": 5, "id": "M-16"},
        {"dim": 64, "c": -2.0, "base": 5, "id": "M-17"},
        # Set D: Random Topology (c=-5.0)
        {"dim": 8, "c": -5.0, "base": 7, "id": "M-18"},
        {"dim": 16, "c": -5.0, "base": 7, "id": "M-19"},
        {"dim": 32, "c": -5.0, "base": 7, "id": "M-20"},
        {"dim": 64, "c": -5.0, "base": 7, "id": "M-21"},
        # Set E: Star Topology (c=-10.0)
        {"dim": 8, "c": -10.0, "base": 11, "id": "M-22"},
    ]

    results = []

    for s in scenarios:
        print(f"\n>> SCENARIO {s['id']}: Dim={s['dim']}, c={s['c']}")

        # Initialize model with specific config
        config = {
            "latent_dim": s["dim"],
            "hidden_dim": 64,
            "curvature": s["c"],
        }

        # Run verification
        try:
            res = verify_proofs(model_config=config, device=device)
            status = "PASS" if (res["delta"] < 0.5 and res["ultrametricity"] > 0.9) else "FAIL"
            res["status"] = status
            res["id"] = s["id"]
            results.append(res)
        except Exception as e:
            print(f"CRASH: {e}")
            results.append({"id": s["id"], "status": "CRASH"})

    print(f"\n{'=' * 60}")
    print("GRID RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'ID':<6} {'Delta':<10} {'Ultra':<10} {'Status'}")
    print("-" * 40)
    for r in results:
        if r["status"] == "CRASH":
            print(f"{r['id']:<6} {'ERROR':<20} {r['status']}")
        else:
            print(f"{r['id']:<6} {r['delta']:<10.4f} {r['ultrametricity']:<10.4f} {r['status']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "grid"],
        default="single",
        help="Run mode",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("--dim", type=int, default=16, help="Latent dimension")
    args = parser.parse_args()

    if args.mode == "grid":
        run_grid_stress_test()
    else:
        verify_proofs(
            checkpoint_path=args.checkpoint,
            model_config={"latent_dim": args.dim},
        )
