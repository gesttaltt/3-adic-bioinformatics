# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Downstream Validation for V5.11.3.

Tests whether learned hyperbolic embeddings are useful for real tasks:
1. Nearest-Neighbor Retrieval: Do neighbors share similar valuations?
2. Hierarchy Preservation: Is radius ordering correct across all points?
3. Arithmetic Structure: Can embeddings predict operation results?

Usage:
    python src/scripts/eval/downstream_validation.py
    python src/scripts/eval/downstream_validation.py --checkpoint path/to/best.pt
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import CHECKPOINTS_DIR
from src.core import TERNARY
from src.data.generation import generate_all_ternary_operations
from src.models import TernaryVAEV5_11_PartialFreeze


def load_model(checkpoint_path: Path, device: str = "cuda"):
    """Load trained model from checkpoint."""
    # Create model with same architecture as training
    # Note: Must match training architecture including dropout (even if disabled via eval mode)
    model = TernaryVAEV5_11_PartialFreeze(
        latent_dim=16,
        hidden_dim=128,
        max_radius=0.95,
        curvature=1.0,
        use_controller=False,
        use_dual_projection=True,
        n_projection_layers=2,
        projection_dropout=0.1,  # Must match training architecture
        freeze_encoder_b=False,
        encoder_b_lr_scale=0.1,
    )

    # Load v5.5 base checkpoint first
    v5_5_path = CHECKPOINTS_DIR / "v5_5" / "latest.pt"
    model.load_v5_5_checkpoint(v5_5_path, device)

    # Load trained weights (projection + encoder_B)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load full model state
    if "model_state" in checkpoint:
        state = checkpoint["model_state"]
    elif "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state = checkpoint

    # Load all trainable weights (projection + encoder_B for Option C)
    trainable_state = {k: v for k, v in state.items() if "projection" in k or "encoder_B" in k}
    if trainable_state:
        model.load_state_dict(trainable_state, strict=False)
        print(f"Loaded {len(trainable_state)} weight tensors from {checkpoint_path}")
        # Show what was loaded
        proj_keys = [k for k in trainable_state if "projection" in k]
        enc_keys = [k for k in trainable_state if "encoder_B" in k]
        print(f"  Projection: {len(proj_keys)} tensors")
        print(f"  Encoder_B: {len(enc_keys)} tensors")

    model.to(device)
    model.eval()
    return model


def get_embeddings(model, device: str = "cuda"):
    """Get hyperbolic embeddings for all operations."""
    operations = generate_all_ternary_operations()
    x = torch.tensor(operations, dtype=torch.float32, device=device)
    indices = torch.arange(len(operations), device=device)

    with torch.no_grad():
        outputs = model(x, compute_control=False)
        z_A_hyp = outputs["z_A_hyp"]
        z_B_hyp = outputs["z_B_hyp"]

    valuations = TERNARY.valuation(indices).cpu().numpy()

    return {
        "z_A": z_A_hyp.cpu().numpy(),
        "z_B": z_B_hyp.cpu().numpy(),
        "valuations": valuations,
        "indices": indices.cpu().numpy(),
        "operations": operations,
    }


def test_nearest_neighbor_retrieval(embeddings: dict, k: int = 10):
    """Test if nearest neighbors share similar valuations.

    Metric: For each point, what fraction of its k-NN have the same or
    adjacent valuation level?
    """
    print("\n" + "=" * 60)
    print("TEST 1: Nearest-Neighbor Retrieval")
    print("=" * 60)

    results = {}

    for name, z in [
        ("VAE-A", embeddings["z_A"]),
        ("VAE-B", embeddings["z_B"]),
    ]:
        valuations = embeddings["valuations"]

        # Fit k-NN
        nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(z)
        distances, indices = nbrs.kneighbors(z)

        # For each point, check neighbor valuations
        same_valuation = 0
        adjacent_valuation = 0  # within ±1
        total = 0

        valuation_agreement = []

        for i in range(len(z)):
            my_val = valuations[i]
            neighbor_vals = valuations[indices[i, 1:]]  # Skip self

            same = np.sum(neighbor_vals == my_val)
            adjacent = np.sum(np.abs(neighbor_vals - my_val) <= 1)

            same_valuation += same
            adjacent_valuation += adjacent
            total += k

            valuation_agreement.append(same / k)

        same_rate = same_valuation / total
        adjacent_rate = adjacent_valuation / total

        # Compute by valuation level
        by_level = defaultdict(list)
        for i, v in enumerate(valuations):
            by_level[int(v)].append(valuation_agreement[i])

        print(f"\n{name}:")
        print(f"  Same valuation rate (k={k}): {same_rate * 100:.1f}%")
        print(f"  Adjacent valuation rate (±1): {adjacent_rate * 100:.1f}%")
        print("  By valuation level:")
        for v in sorted(by_level.keys()):
            rates = by_level[v]
            print(f"    v={v}: {np.mean(rates) * 100:.1f}% same (n={len(rates)})")

        results[name] = {
            "same_rate": same_rate,
            "adjacent_rate": adjacent_rate,
            "by_level": {v: np.mean(r) for v, r in by_level.items()},
        }

    return results


def test_hierarchy_preservation(embeddings: dict):
    """Test if radius ordering matches valuation ordering.

    Metrics:
    - Spearman correlation between valuation and radius
    - Kendall's tau for rank agreement
    - Pairwise accuracy: fraction of pairs with correct ordering
    """
    print("\n" + "=" * 60)
    print("TEST 2: Hierarchy Preservation")
    print("=" * 60)

    results = {}

    for name, z in [
        ("VAE-A", embeddings["z_A"]),
        ("VAE-B", embeddings["z_B"]),
    ]:
        valuations = embeddings["valuations"]
        radii = np.linalg.norm(z, axis=1)

        # Correlations (should be negative: high valuation = small radius)
        spearman_corr, spearman_p = spearmanr(valuations, radii)
        kendall_corr, kendall_p = kendalltau(valuations, radii)

        # Pairwise accuracy (sample pairs for efficiency)
        n_pairs = 50000
        i_idx = np.random.randint(0, len(z), n_pairs)
        j_idx = np.random.randint(0, len(z), n_pairs)

        # Filter to pairs with different valuations
        diff_mask = valuations[i_idx] != valuations[j_idx]
        i_idx = i_idx[diff_mask]
        j_idx = j_idx[diff_mask]

        v_i, v_j = valuations[i_idx], valuations[j_idx]
        r_i, r_j = radii[i_idx], radii[j_idx]

        # Correct ordering: higher valuation should have smaller radius
        # If v_i > v_j, then r_i should be < r_j
        correct = ((v_i > v_j) & (r_i < r_j)) | ((v_i < v_j) & (r_i > r_j))
        pairwise_accuracy = np.mean(correct)

        # Radius statistics by valuation
        print(f"\n{name}:")
        print(f"  Spearman correlation: {spearman_corr:.4f} (p={spearman_p:.2e})")
        print(f"  Kendall's tau: {kendall_corr:.4f} (p={kendall_p:.2e})")
        print(f"  Pairwise accuracy: {pairwise_accuracy * 100:.1f}%")
        print("  Radius by valuation level:")

        for v in range(10):
            mask = valuations == v
            if mask.any():
                r_mean = radii[mask].mean()
                r_std = radii[mask].std()
                print(f"    v={v}: radius={r_mean:.3f} ± {r_std:.3f} (n={mask.sum()})")

        results[name] = {
            "spearman": spearman_corr,
            "kendall": kendall_corr,
            "pairwise_accuracy": pairwise_accuracy,
        }

    return results


def test_arithmetic_structure(embeddings: dict):
    """Test if embeddings capture arithmetic structure.

    Task: Given embeddings of a, b, c in operation (a, b, c) -> result,
    can we predict properties of the result?

    Tests:
    - Predict result valuation from embedding
    - Predict if result is zero
    """
    print("\n" + "=" * 60)
    print("TEST 3: Arithmetic Structure")
    print("=" * 60)

    results = {}
    operations = embeddings["operations"]
    valuations = embeddings["valuations"]

    for name, z in [
        ("VAE-A", embeddings["z_A"]),
        ("VAE-B", embeddings["z_B"]),
    ]:
        # Task 1: Predict valuation level (multi-class classification)
        # Bin valuations into 3 classes: low (0-2), mid (3-5), high (6-9)
        val_classes = np.zeros(len(valuations), dtype=int)
        val_classes[valuations <= 2] = 0
        val_classes[(valuations > 2) & (valuations <= 5)] = 1
        val_classes[valuations > 5] = 2

        clf = LogisticRegression(max_iter=1000, random_state=42)
        val_scores = cross_val_score(clf, z, val_classes, cv=5)
        val_accuracy = val_scores.mean()

        # Task 2: Predict if operation contains zero result component
        # Check if any component of the result is 0
        has_zero = np.array([0 in op[6:9] for op in operations])

        clf_zero = LogisticRegression(max_iter=1000, random_state=42)
        zero_scores = cross_val_score(clf_zero, z, has_zero.astype(int), cv=5)
        zero_accuracy = zero_scores.mean()

        # Task 3: Predict specific result component
        # Predict the first component of the result (-1, 0, or 1)
        result_c0 = np.array([op[6] for op in operations])  # First result component
        result_c0_class = result_c0 + 1  # Map to 0, 1, 2

        clf_c0 = LogisticRegression(max_iter=1000, random_state=42, multi_class="multinomial")
        c0_scores = cross_val_score(clf_c0, z, result_c0_class, cv=5)
        c0_accuracy = c0_scores.mean()

        print(f"\n{name}:")
        print(f"  Valuation class prediction (3-class): {val_accuracy * 100:.1f}% (baseline: 33.3%)")
        print(f"  Has-zero prediction (binary): {zero_accuracy * 100:.1f}% (baseline: ~50%)")
        print(f"  Result component prediction (3-class): {c0_accuracy * 100:.1f}% (baseline: 33.3%)")

        results[name] = {
            "valuation_accuracy": val_accuracy,
            "has_zero_accuracy": zero_accuracy,
            "result_component_accuracy": c0_accuracy,
        }

    return results


def compute_production_readiness_score(nn_results, hierarchy_results, arithmetic_results):
    """Compute overall production readiness score."""
    print("\n" + "=" * 60)
    print("PRODUCTION READINESS ASSESSMENT")
    print("=" * 60)

    # Thresholds for production readiness
    thresholds = {
        "nn_adjacent_rate": 0.70,  # 70% neighbors within ±1 valuation
        "hierarchy_pairwise": 0.85,  # 85% pairwise ordering correct
        "hierarchy_spearman": -0.65,  # Strong negative correlation
        "arithmetic_valuation": 0.50,  # Better than random (33%)
    }

    # Check VAE-A (the one we optimized)
    vae_a_nn = nn_results["VAE-A"]["adjacent_rate"]
    vae_a_pairwise = hierarchy_results["VAE-A"]["pairwise_accuracy"]
    vae_a_spearman = hierarchy_results["VAE-A"]["spearman"]
    vae_a_arith = arithmetic_results["VAE-A"]["valuation_accuracy"]

    checks = {
        "NN Adjacent Rate": (
            vae_a_nn >= thresholds["nn_adjacent_rate"],
            vae_a_nn,
            thresholds["nn_adjacent_rate"],
        ),
        "Hierarchy Pairwise": (
            vae_a_pairwise >= thresholds["hierarchy_pairwise"],
            vae_a_pairwise,
            thresholds["hierarchy_pairwise"],
        ),
        "Hierarchy Spearman": (
            vae_a_spearman <= thresholds["hierarchy_spearman"],
            vae_a_spearman,
            thresholds["hierarchy_spearman"],
        ),
        "Arithmetic Prediction": (
            vae_a_arith >= thresholds["arithmetic_valuation"],
            vae_a_arith,
            thresholds["arithmetic_valuation"],
        ),
    }

    passed = sum(1 for v in checks.values() if v[0])
    total = len(checks)

    print(f"\nVAE-A Assessment ({passed}/{total} checks passed):")
    for name, (check_passed, value, threshold) in checks.items():
        status = "PASS" if check_passed else "FAIL"
        print(f"  [{status}] {name}: {value:.3f} (threshold: {threshold:.3f})")

    # Overall verdict
    print("\n" + "-" * 60)
    if passed == total:
        print("VERDICT: PRODUCTION READY")
        print("All downstream validation checks passed.")
    elif passed >= total - 1:
        print("VERDICT: NEAR PRODUCTION READY")
        print("Minor improvements may help, but current quality is acceptable.")
    else:
        print("VERDICT: NEEDS IMPROVEMENT")
        print("Consider option (2): unfreezing VAE-A encoder for better hierarchy.")

    return passed, total


def main():
    parser = argparse.ArgumentParser(description="Downstream Validation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(CHECKPOINTS_DIR / "ternary_option_c_dual" / "best.pt"),
        help="Path to model checkpoint",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--k", type=int, default=10, help="Number of neighbors for NN test")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    checkpoint_path = PROJECT_ROOT / args.checkpoint
    if not checkpoint_path.exists():
        # Try to find latest checkpoint
        alt_paths = [
            CHECKPOINTS_DIR / "ternary_option_c_dual" / "best.pt",
            CHECKPOINTS_DIR / "ternary_option_c_dual" / "latest.pt",
        ]
        for alt in alt_paths:
            if alt.exists():
                checkpoint_path = alt
                break
        else:
            print(f"ERROR: No checkpoint found at {checkpoint_path}")
            print("Please specify a valid checkpoint path with --checkpoint")
            sys.exit(1)

    print(f"\nLoading model from: {checkpoint_path}")
    model = load_model(checkpoint_path, device)

    # Get embeddings
    print("\nGenerating embeddings for all 19,683 operations...")
    embeddings = get_embeddings(model, device)

    # Run tests
    nn_results = test_nearest_neighbor_retrieval(embeddings, k=args.k)
    hierarchy_results = test_hierarchy_preservation(embeddings)
    arithmetic_results = test_arithmetic_structure(embeddings)

    # Production readiness assessment
    passed, total = compute_production_readiness_score(nn_results, hierarchy_results, arithmetic_results)

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
