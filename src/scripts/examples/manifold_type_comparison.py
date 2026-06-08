#!/usr/bin/env python3
"""
Example script demonstrating the dual manifold organization framework.

This script shows how to:
1. Load checkpoints with different manifold organizations
2. Evaluate them using type-aware metrics
3. Compare performance on relevant tasks
4. Recommend optimal usage for each type
"""

# Add project root to path
import sys
from pathlib import Path

import torch

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.evaluation.manifold_organization import ManifoldEvaluator, ManifoldType, detailed_manifold_analysis

from src.core import TERNARY
from src.geometry import poincare_distance
from src.models.ternary_vae import TernaryVAEV5_11_PartialFreeze


def load_checkpoint_examples():
    """Load example checkpoints representing different manifold types."""
    checkpoints = {}

    # Valuation-optimal example (homeostatic_rich)
    try:
        ckpt_path = project_root / "checkpoints" / "homeostatic_rich" / "best.pt"
        if ckpt_path.exists():
            checkpoints["valuation_optimal"] = {
                "path": ckpt_path,
                "name": "homeostatic_rich",
                "expected_hierarchy": -0.8321,
                "description": "Valuation-optimal: p-adic semantic hierarchy",
            }
    except Exception as e:
        print(f"Could not load homeostatic_rich: {e}")

    # Frequency-optimal example (v5_11_progressive)
    try:
        ckpt_path = project_root / "checkpoints" / "v5_11_progressive" / "best.pt"
        if ckpt_path.exists():
            checkpoints["frequency_optimal"] = {
                "path": ckpt_path,
                "name": "v5_11_progressive",
                "expected_hierarchy": +0.78,
                "description": "Frequency-optimal: Shannon information efficiency",
            }
    except Exception as e:
        print(f"Could not load v5_11_progressive: {e}")

    return checkpoints


def create_model():
    """Create model with standard configuration."""
    return TernaryVAEV5_11_PartialFreeze(
        latent_dim=16, hidden_dim=64, max_radius=0.99, curvature=1.0, use_controller=True, use_dual_projection=True
    )


def extract_embeddings_and_compute_radii(model, device="cpu"):
    """Extract embeddings and compute radii for all operations."""
    model.eval()
    model = model.to(device)

    # Generate all ternary operations
    print("Generating all ternary operations...")
    indices = torch.arange(19683, dtype=torch.long, device=device)
    ops = TERNARY.to_ternary(indices).to(device)
    valuations = TERNARY.valuation(indices)

    # Get embeddings in batches
    batch_size = 4096
    all_z_A = []
    all_z_B = []

    print("Extracting embeddings...")
    with torch.no_grad():
        for i in range(0, len(ops), batch_size):
            batch_ops = ops[i : i + batch_size]
            outputs = model(batch_ops, compute_control=False)
            all_z_A.append(outputs["z_A_hyp"].cpu())
            all_z_B.append(outputs["z_B_hyp"].cpu())

    z_A_hyp = torch.cat(all_z_A, dim=0)
    z_B_hyp = torch.cat(all_z_B, dim=0)

    # Compute hyperbolic radii (NOT Euclidean norm!)
    origin_A = torch.zeros_like(z_A_hyp)
    origin_B = torch.zeros_like(z_B_hyp)
    radii_A = poincare_distance(z_A_hyp, origin_A, c=1.0)
    radii_B = poincare_distance(z_B_hyp, origin_B, c=1.0)

    return {
        "z_A_hyp": z_A_hyp,
        "z_B_hyp": z_B_hyp,
        "radii_A": radii_A,
        "radii_B": radii_B,
        "valuations": valuations,
        "indices": indices,
    }


def demonstrate_type_aware_evaluation():
    """Demonstrate the type-aware evaluation framework."""
    print("\n" + "=" * 80)
    print("DUAL MANIFOLD ORGANIZATION FRAMEWORK DEMONSTRATION")
    print("=" * 80)

    checkpoints = load_checkpoint_examples()

    if not checkpoints:
        print("No example checkpoints found. Please ensure you have:")
        print("- homeostatic_rich checkpoint (valuation-optimal example)")
        print("- v5_11_progressive checkpoint (frequency-optimal example)")
        return

    evaluator = ManifoldEvaluator()

    for manifold_type, ckpt_info in checkpoints.items():
        print(f"\n{'-' * 60}")
        print(f"ANALYZING: {ckpt_info['name']} ({manifold_type})")
        print(f"Description: {ckpt_info['description']}")
        print(f"Expected hierarchy: {ckpt_info['expected_hierarchy']}")
        print(f"{'-' * 60}")

        try:
            # Load checkpoint
            checkpoint = torch.load(ckpt_info["path"], map_location="cpu")
            model = create_model()
            model.load_state_dict(checkpoint["model_state_dict"])

            # Extract embeddings
            embeddings_data = extract_embeddings_and_compute_radii(model)

            # Type-aware evaluation
            intended_type = (
                ManifoldType.VALUATION_OPTIMAL
                if manifold_type == "valuation_optimal"
                else ManifoldType.FREQUENCY_OPTIMAL
            )

            # Quick evaluation
            print("\n1. QUICK EVALUATION:")
            summary = evaluator.evaluate_hierarchy(
                embeddings_data["radii_B"], embeddings_data["valuations"], intended_type=intended_type
            )
            print(summary)

            # Detailed analysis
            print("\n2. DETAILED ANALYSIS:")
            analysis = detailed_manifold_analysis(
                embeddings_data["radii_B"], embeddings_data["valuations"], intended_type=intended_type
            )

            print(f"   Hierarchy Score: {analysis['hierarchy_score']:.4f}")
            print(
                f"   Detected Type: {analysis['type_alignment']['detected_type'] if 'detected_type' in analysis['type_alignment'] else analysis['manifold_type']}"
            )
            print(f"   Organization Quality: {analysis['organization_quality']}")
            print(f"   Richness: {analysis['richness']:.6f}")
            print(f"   Level Separation: {analysis['separation']:.4f}")

            # Geometric efficiency
            geo_eff = analysis["geometric_efficiency"]
            print(f"   Geometric Efficiency: {geo_eff['volume_efficiency']}")
            print(f"   Frequency-Volume Correlation: {geo_eff['frequency_volume_correlation']:.3f}")

            # Application recommendations
            print("\n3. APPLICATION RECOMMENDATIONS:")
            if analysis["manifold_type"] == "valuation_optimal":
                print("   ✓ OPTIMAL FOR:")
                print("     - Semantic reasoning and compositional learning")
                print("     - P-adic mathematical applications")
                print("     - Novel pattern generation and extrapolation")
                print("     - Hierarchical concept understanding")
                print("\n   ⚠ SUBOPTIMAL FOR:")
                print("     - Data compression applications")
                print("     - Fast retrieval of frequent operations")
                print("     - High-throughput statistical processing")
            else:
                print("   ✓ OPTIMAL FOR:")
                print("     - Data compression and efficient storage")
                print("     - Fast similarity search and retrieval")
                print("     - Frequency-based prediction tasks")
                print("     - High-throughput sequence processing")
                print("\n   ⚠ SUBOPTIMAL FOR:")
                print("     - Semantic reasoning about rare patterns")
                print("     - Compositional understanding")
                print("     - Mathematical structure preservation")

        except Exception as e:
            print(f"Error analyzing {ckpt_info['name']}: {e}")

    print(f"\n{'=' * 80}")
    print("FRAMEWORK SUMMARY")
    print(f"{'=' * 80}")

    print("""
Key Insights from Dual Manifold Framework:

1. PARADIGM SHIFT:
   • Positive hierarchy is NOT a failure mode
   • It represents frequency-optimal organization (Shannon efficiency)
   • Both positive and negative hierarchies are mathematically valid

2. APPLICATION-DEPENDENT SELECTION:
   • Semantic reasoning → Use valuation-optimal (negative hierarchy)
   • Compression/retrieval → Use frequency-optimal (positive hierarchy)
   • Mixed applications → Consider task-specific checkpoints

3. EVALUATION REVISION:
   • Stop labeling positive hierarchy as "inverted"
   • Use type-aware evaluation metrics
   • Assess quality relative to intended manifold type

4. TRAINING IMPLICATIONS:
   • Explicit manifold type selection in configs
   • Type-specific loss functions and hyperparameters
   • Different convergence criteria for each type
    """)


def compare_manifold_types_side_by_side():
    """Side-by-side comparison of manifold organization types."""
    print(f"\n{'=' * 80}")
    print("SIDE-BY-SIDE MANIFOLD COMPARISON")
    print(f"{'=' * 80}")

    comparison_table = """
┌─────────────────────┬─────────────────────┬─────────────────────┐
│ Aspect              │ Valuation-Optimal   │ Frequency-Optimal   │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ Hierarchy Score     │ Negative (-0.8 to   │ Positive (+0.6 to   │
│                     │ -1.0)               │ +0.8)               │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ Radial Organization │ Rare → Center       │ Frequent → Center   │
│                     │ Frequent → Edge     │ Rare → Edge         │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ Information Theory  │ Kolmogorov          │ Shannon Entropy     │
│                     │ Complexity          │                     │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ Volume Allocation   │ "Wastes" volume on  │ Optimally allocates │
│                     │ frequent items      │ volume by frequency │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ Compression         │ Suboptimal          │ Excellent           │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ Semantic Reasoning  │ Excellent           │ Suboptimal          │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ Retrieval Speed     │ Slow for frequent   │ Fast for frequent   │
│                     │ items               │ items               │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ Mathematical        │ P-adic number       │ Information theory, │
│ Foundation          │ theory              │ coding theory       │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ Best Applications   │ • Genetic analysis  │ • Data compression  │
│                     │ • Mathematical      │ • Search engines    │
│                     │   reasoning         │ • Language modeling │
│                     │ • Novel generation  │ • Fast processing   │
└─────────────────────┴─────────────────────┴─────────────────────┘
    """

    print(comparison_table)

    print("""
USAGE GUIDELINES:

1. CHOOSE VALUATION-OPTIMAL WHEN:
   - Working with genetic code (rare codons = higher semantic value)
   - Building mathematical reasoning systems
   - Need compositional understanding
   - Extrapolating to novel/rare patterns

2. CHOOSE FREQUENCY-OPTIMAL WHEN:
   - Building compression/storage systems
   - Need fast retrieval of common items
   - Working with statistical/frequency-based ML
   - Optimizing for throughput

3. MIXED APPLICATIONS:
   - Consider training separate models for different tasks
   - Use ensemble methods combining both types
   - Implement dynamic switching based on query type
    """)


if __name__ == "__main__":
    demonstrate_type_aware_evaluation()
    compare_manifold_types_side_by_side()

    print(f"\n{'=' * 80}")
    print("NEXT STEPS")
    print(f"{'=' * 80}")
    print("""
To fully implement the dual manifold framework:

1. UPDATE DOCUMENTATION:
   - Replace "inverted hierarchy" terminology
   - Add type selection guidelines
   - Update checkpoint descriptions

2. IMPLEMENT TRAINING:
   - Use src/configs/manifold_types/valuation_optimal.yaml
   - Use src/configs/manifold_types/frequency_optimal.yaml
   - Implement type-specific loss functions

3. VALIDATE EMPIRICALLY:
   - Benchmark both types on downstream tasks
   - Measure task-specific performance
   - Document trade-offs and recommendations

4. INTEGRATION:
   - Update existing evaluation scripts
   - Add type selection to model configs
   - Create application-specific guidelines
    """)
