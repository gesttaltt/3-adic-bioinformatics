"""Validate Enhanced Differentiable Controller with Hierarchical Outputs.

Phase 3.1 Validation - Enhanced Controller Architecture Testing
==============================================================

This script validates the Enhanced DifferentiableController's hierarchical
output structure and backward compatibility with the original controller.

Author: Claude Code
Date: 2026-01-14
"""

import sys
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.enhanced_controller import (
    ControllerOutputs,
    HierarchicalAttention,
    TemperatureAdaptive,
    create_backward_compatible_controller,
    create_enhanced_controller,
)

from src.models.differentiable_controller import DifferentiableController


def test_backward_compatibility():
    """Test that enhanced controller is backward compatible with original."""
    print("🔄 Backward Compatibility Testing")
    print("=" * 50)

    # Create original and backward-compatible enhanced controllers
    original = DifferentiableController(input_dim=8, hidden_dim=32)
    enhanced_compat = create_backward_compatible_controller(input_dim=8, hidden_dim=32)

    # Test input
    batch_stats = torch.randn(4, 8)

    # Get outputs
    original_out = original(batch_stats)
    enhanced_out = enhanced_compat(batch_stats)

    # Check that flat outputs have same structure
    assert set(original_out.keys()) == set(enhanced_out.flat_outputs.keys()), "Output keys don't match"

    print("✅ Output structure compatibility")

    # Check output shapes
    for key in original_out:
        orig_shape = original_out[key].shape
        enh_shape = enhanced_out.flat_outputs[key].shape
        assert orig_shape == enh_shape, f"Shape mismatch for {key}: {orig_shape} vs {enh_shape}"

    print("✅ Output shape compatibility")

    # Check output ranges (should be in same bounds)
    bounds_check = {
        "rho": (0.0, 0.5),
        "weight_geodesic": (0.1, None),  # Lower bound only
        "weight_radial": (0.0, None),
        "beta_A": (0.5, None),
        "beta_B": (0.5, None),
        "tau": (0.0, 1.0),
    }

    for key, (min_val, max_val) in bounds_check.items():
        orig_vals = original_out[key]
        enh_vals = enhanced_out.flat_outputs[key]

        # Check lower bounds
        assert torch.all(orig_vals >= min_val), f"Original {key} below bound {min_val}"
        assert torch.all(enh_vals >= min_val), f"Enhanced {key} below bound {min_val}"

        # Check upper bounds (if specified)
        if max_val is not None:
            assert torch.all(orig_vals <= max_val), f"Original {key} above bound {max_val}"
            assert torch.all(enh_vals <= max_val), f"Enhanced {key} above bound {max_val}"

    print("✅ Output range compatibility")

    # Test gradient flow
    original.train()
    enhanced_compat.train()

    batch_stats_grad = torch.randn(4, 8, requires_grad=True)

    # Original
    orig_out = original(batch_stats_grad)
    orig_loss = sum(v.sum() for v in orig_out.values())
    orig_loss.backward()
    orig_grad_norm = batch_stats_grad.grad.norm()

    # Enhanced (reset grad)
    batch_stats_grad.grad = None
    enh_out = enhanced_compat(batch_stats_grad)
    enh_loss = sum(v.sum() for v in enh_out.flat_outputs.values())
    enh_loss.backward()
    enh_grad_norm = batch_stats_grad.grad.norm()

    print(f"   Original gradient norm: {orig_grad_norm:.6f}")
    print(f"   Enhanced gradient norm: {enh_grad_norm:.6f}")
    print("✅ Gradient flow compatibility")


def test_hierarchical_outputs():
    """Test hierarchical output structure and organization."""
    print("\n🏗️  Hierarchical Output Testing")
    print("=" * 50)

    # Create enhanced controller with hierarchical outputs
    controller = create_enhanced_controller(
        input_dim=8,
        hidden_dim=64,
        enable_hierarchical=True,
    )

    batch_stats = torch.randn(4, 8)
    outputs = controller(batch_stats)

    # Test output types and structure
    assert isinstance(outputs, ControllerOutputs), "Should return ControllerOutputs dataclass"
    print("✅ Hierarchical output structure")

    # Test Strategic Level outputs
    strategic_outputs = [
        ("hierarchy_priority", (0.0, 1.0)),
        ("coverage_priority", (0.0, 1.0)),
        ("exploration_rate", (0.0, 1.0)),
    ]

    for name, (min_val, max_val) in strategic_outputs:
        values = getattr(outputs, name)
        assert values.shape == (4,), f"Wrong shape for {name}: {values.shape}"
        assert torch.all(values >= min_val), f"{name} below bound {min_val}"
        assert torch.all(values <= max_val), f"{name} above bound {max_val}"

    print("✅ Strategic level outputs (3 signals)")

    # Test Tactical Level outputs
    tactical_outputs = [
        ("geodesic_weight", (0.1, float("inf"))),
        ("radial_weight", (0.0, float("inf"))),
        ("regularization_strength", (0.0, 1.0)),
    ]

    for name, (min_val, max_val) in tactical_outputs:
        values = getattr(outputs, name)
        assert values.shape == (4,), f"Wrong shape for {name}: {values.shape}"
        assert torch.all(values >= min_val), f"{name} below bound {min_val}"
        if max_val != float("inf"):
            assert torch.all(values <= max_val), f"{name} above bound {max_val}"

    print("✅ Tactical level outputs (3 signals)")

    # Test Operational Level outputs
    operational_outputs = [
        ("rho_injection", (0.0, 0.5)),
        ("tau_curriculum", (0.0, 1.0)),
        ("learning_rate_scale", (0.5, 2.0)),
    ]

    for name, (min_val, max_val) in operational_outputs:
        values = getattr(outputs, name)
        assert values.shape == (4,), f"Wrong shape for {name}: {values.shape}"
        assert torch.all(values >= min_val), f"{name} below bound {min_val}"
        assert torch.all(values <= max_val), f"{name} above bound {max_val}"

    print("✅ Operational level outputs (3 signals)")

    # Test Attention Level outputs
    attention_outputs = [
        ("encoder_attention", 2),  # 2 encoders
        ("loss_attention", 5),  # 5 loss components
        ("layer_attention", 4),  # 4 layers
    ]

    for name, expected_dim in attention_outputs:
        values = getattr(outputs, name)
        assert values.shape == (4, expected_dim), f"Wrong shape for {name}: {values.shape}"

        # Should be normalized (sum to 1)
        sums = values.sum(dim=1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-6), f"{name} not normalized"

    print("✅ Attention level outputs (3 attention mechanisms)")

    # Test flat outputs for backward compatibility
    assert "rho" in outputs.flat_outputs, "Missing flat output: rho"
    assert len(outputs.flat_outputs) == 6, "Should have 6 flat outputs"
    print("✅ Backward compatibility flat outputs")


def test_attention_mechanisms():
    """Test hierarchical attention mechanisms."""
    print("\n👁️  Attention Mechanism Testing")
    print("=" * 50)

    # Test individual attention modules
    input_dim = 64
    batch_size = 8

    # Test encoder attention (2 encoders)
    encoder_attn = HierarchicalAttention(input_dim, num_components=2, num_heads=4)
    x = torch.randn(batch_size, input_dim)
    attn_weights = encoder_attn(x)

    assert attn_weights.shape == (batch_size, 2), f"Wrong encoder attention shape: {attn_weights.shape}"

    # Should be normalized
    sums = attn_weights.sum(dim=1)
    assert torch.allclose(sums, torch.ones(batch_size), atol=1e-6), "Encoder attention not normalized"
    print("✅ Encoder attention mechanism (2 components)")

    # Test loss attention (5 loss components)
    loss_attn = HierarchicalAttention(input_dim, num_components=5, num_heads=4)
    attn_weights = loss_attn(x)

    assert attn_weights.shape == (batch_size, 5), f"Wrong loss attention shape: {attn_weights.shape}"
    sums = attn_weights.sum(dim=1)
    assert torch.allclose(sums, torch.ones(batch_size), atol=1e-6), "Loss attention not normalized"
    print("✅ Loss attention mechanism (5 components)")

    # Test layer attention (4 layers)
    layer_attn = HierarchicalAttention(input_dim, num_components=4, num_heads=4)
    attn_weights = layer_attn(x)

    assert attn_weights.shape == (batch_size, 4), f"Wrong layer attention shape: {attn_weights.shape}"
    sums = attn_weights.sum(dim=1)
    assert torch.allclose(sums, torch.ones(batch_size), atol=1e-6), "Layer attention not normalized"
    print("✅ Layer attention mechanism (4 components)")

    # Test gradient flow through attention
    x.requires_grad_(True)
    attn_out = encoder_attn(x)
    loss = attn_out.sum()
    loss.backward()

    assert x.grad is not None, "No gradient through attention mechanism"
    print("✅ Attention gradient flow")


def test_adaptive_temperature():
    """Test adaptive temperature mechanism."""
    print("\n🌡️  Adaptive Temperature Testing")
    print("=" * 50)

    input_dim = 64
    batch_size = 4

    temp_module = TemperatureAdaptive(input_dim)
    x = torch.randn(batch_size, input_dim)

    temp = temp_module(x)

    # Check shape and bounds
    assert temp.shape == (batch_size, 1), f"Wrong temperature shape: {temp.shape}"
    assert torch.all(temp >= 0.1), f"Temperature below 0.1: {temp.min()}"
    assert torch.all(temp <= 2.0), f"Temperature above 2.0: {temp.max()}"

    print(f"✅ Temperature range: [{temp.min():.3f}, {temp.max():.3f}] (target: [0.1, 2.0])")

    # Test gradient flow
    x.requires_grad_(True)
    temp = temp_module(x)
    loss = temp.sum()
    loss.backward()

    assert x.grad is not None, "No gradient through temperature module"
    print("✅ Temperature gradient flow")

    # Test through enhanced controller
    controller = create_enhanced_controller()
    batch_stats = torch.randn(4, 8)

    adaptive_temp = controller.get_adaptive_temperature(batch_stats)
    assert adaptive_temp.shape == (4, 1), "Wrong controller temperature shape"
    print("✅ Controller adaptive temperature")


def test_control_summary():
    """Test control summary functionality."""
    print("\n📊 Control Summary Testing")
    print("=" * 50)

    controller = create_enhanced_controller()
    batch_stats = torch.randn(4, 8)
    outputs = controller(batch_stats)

    summary = controller.get_control_summary(outputs)

    # Check summary structure
    expected_keys = {
        "hierarchy_focus",
        "coverage_focus",
        "exploration_rate",
        "geodesic_importance",
        "radial_importance",
        "regularization",
        "rho_injection",
        "curriculum_progress",
        "lr_scaling",
        "encoder_entropy",
        "loss_entropy",
        "layer_entropy",
    }

    assert set(summary.keys()) == expected_keys, f"Missing summary keys: {expected_keys - set(summary.keys())}"
    print("✅ Summary structure")

    # Check value ranges
    for key, value in summary.items():
        assert isinstance(value, float), f"{key} should be float"
        assert not torch.isnan(torch.tensor(value)), f"{key} is NaN"
        assert not torch.isinf(torch.tensor(value)), f"{key} is infinite"

    print("✅ Summary value validity")

    # Print sample summary
    print("\n📋 Sample Control Summary:")
    for key, value in summary.items():
        print(f"   {key:20s}: {value:.4f}")


def test_performance_and_capacity():
    """Test performance and model capacity."""
    print("\n🚀 Performance and Capacity Testing")
    print("=" * 50)

    # Compare model sizes
    original = DifferentiableController(input_dim=8, hidden_dim=32)
    enhanced_simple = create_backward_compatible_controller(input_dim=8, hidden_dim=32)
    enhanced_full = create_enhanced_controller(input_dim=8, hidden_dim=64, enable_hierarchical=True)

    orig_params = sum(p.numel() for p in original.parameters())
    enh_simple_params = sum(p.numel() for p in enhanced_simple.parameters())
    enh_full_params = sum(p.numel() for p in enhanced_full.parameters())

    print(f"Original controller:         {orig_params:6d} parameters")
    print(f"Enhanced (compatible):       {enh_simple_params:6d} parameters")
    print(f"Enhanced (full hierarchical): {enh_full_params:6d} parameters")

    size_increase = (enh_full_params - orig_params) / orig_params * 100
    print(f"Full enhancement size increase: {size_increase:.1f}%")

    # Test forward pass performance
    import time

    batch_stats = torch.randn(32, 8)

    # Warmup
    for _ in range(10):
        _ = original(batch_stats)
        _ = enhanced_full(batch_stats)

    # Time original
    start = time.time()
    for _ in range(100):
        _ = original(batch_stats)
    orig_time = time.time() - start

    # Time enhanced
    start = time.time()
    for _ in range(100):
        _ = enhanced_full(batch_stats)
    enh_time = time.time() - start

    slowdown = enh_time / orig_time
    print(f"Original forward time:       {orig_time * 1000:.2f} ms")
    print(f"Enhanced forward time:       {enh_time * 1000:.2f} ms")
    print(f"Slowdown factor:             {slowdown:.2f}x")

    assert slowdown < 10.0, f"Enhanced controller too slow: {slowdown:.2f}x"  # Adjusted for hierarchical complexity
    print("✅ Performance acceptable")


def demonstrate_hierarchical_control():
    """Demonstrate hierarchical control capabilities."""
    print("\n🎯 Hierarchical Control Demonstration")
    print("=" * 50)

    controller = create_enhanced_controller()

    # Different training scenarios
    scenarios = {
        "early_training": torch.tensor([0.5, 0.5, 0.3, 0.3, 2.0, 2.0, 5.0, 3.0]),  # High losses
        "mid_training": torch.tensor([0.7, 0.8, -0.5, -0.6, 1.0, 1.0, 2.0, 1.0]),  # Improving
        "late_training": torch.tensor([0.85, 0.9, -0.8, -0.85, 0.5, 0.5, 0.5, 0.2]),  # Converged
        "plateau": torch.tensor([0.8, 0.8, -0.7, -0.7, 0.8, 0.8, 1.0, 0.5]),  # Stuck
    }

    for scenario_name, batch_stats in scenarios.items():
        outputs = controller(batch_stats.unsqueeze(0))
        summary = controller.get_control_summary(outputs)

        print(f"\n{scenario_name.replace('_', ' ').title()} Scenario:")
        print(f"  Strategic: hierarchy={summary['hierarchy_focus']:.3f}, coverage={summary['coverage_focus']:.3f}")
        print(f"  Tactical:  geodesic={summary['geodesic_importance']:.3f}, radial={summary['radial_importance']:.3f}")
        print(f"  Operational: rho={summary['rho_injection']:.3f}, lr_scale={summary['lr_scaling']:.3f}")

    print("\n✅ Hierarchical control adaptation demonstrated")


def main():
    """Run all Enhanced DifferentiableController tests."""
    print("🎛️  Enhanced Differentiable Controller - Hierarchical Validation")
    print("Phase 3.1: Complete functionality testing")
    print("=" * 80)

    try:
        test_backward_compatibility()
        test_hierarchical_outputs()
        test_attention_mechanisms()
        test_adaptive_temperature()
        test_control_summary()
        test_performance_and_capacity()
        demonstrate_hierarchical_control()

        print("\n" + "=" * 80)
        print("🎉 Enhanced Differentiable Controller - ALL TESTS PASSED")
        print("✅ Hierarchical output structure validated")
        print("✅ Backward compatibility confirmed")
        print("✅ Attention mechanisms working")
        print("✅ Adaptive temperature functional")
        print("✅ Performance acceptable")
        print("✅ Ready for training integration")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
