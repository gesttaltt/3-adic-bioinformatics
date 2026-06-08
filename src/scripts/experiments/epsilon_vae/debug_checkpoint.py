#!/usr/bin/env python3
"""Debug checkpoint loading - find the architecture mismatch."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import TernaryVAEV5_11_PartialFreeze
from src.utils.checkpoint import get_model_state_dict, load_checkpoint_compat


def main():
    device = "cpu"

    ckpt_path = PROJECT_ROOT / "checkpoints/v5_11_homeostasis/best.pt"
    print(f"Checkpoint: {ckpt_path}")

    # Load checkpoint
    ckpt = load_checkpoint_compat(ckpt_path, map_location=device)

    # Check what key is used for model state
    print(f"\nCheckpoint keys: {list(ckpt.keys())}")

    # Get model state dict
    model_state = get_model_state_dict(ckpt)
    print(f"\nModel state keys: {list(model_state.keys())[:20]}...")
    print(f"Total keys: {len(model_state)}")

    # Check config if present
    if "config" in ckpt:
        print(f"\nStored config: {ckpt['config']}")

    # Create model and check expected keys
    model = TernaryVAEV5_11_PartialFreeze(
        latent_dim=16,
        hidden_dim=64,
        max_radius=0.99,
        curvature=2.0,
        use_controller=False,
        use_dual_projection=True,
        freeze_encoder_b=False,
    )

    expected_keys = set(model.state_dict().keys())
    loaded_keys = set(model_state.keys())

    print(f"\nExpected keys: {len(expected_keys)}")
    print(f"Loaded keys: {len(loaded_keys)}")

    missing = expected_keys - loaded_keys
    extra = loaded_keys - expected_keys

    if missing:
        print(f"\nMissing keys ({len(missing)}):")
        for k in sorted(missing)[:10]:
            print(f"  {k}")

    if extra:
        print(f"\nExtra keys ({len(extra)}):")
        for k in sorted(extra)[:10]:
            print(f"  {k}")

    # Try loading with strict=True to see errors
    print("\n=== Attempting strict load ===")
    try:
        model.load_state_dict(model_state, strict=True)
        print("Strict load succeeded!")
    except Exception as e:
        print(f"Strict load failed: {e}")

    # Check if there's hidden_dim mismatch
    print("\n=== Checking dimensions ===")
    for key in ["encoder_A.layers.0.weight", "encoder_B.layers.0.weight", "decoder_A.layers.0.weight"]:
        if key in model_state:
            print(f"Loaded {key}: {model_state[key].shape}")

    for key in ["encoder_A.layers.0.weight", "encoder_B.layers.0.weight", "decoder_A.layers.0.weight"]:
        if key in model.state_dict():
            print(f"Expected {key}: {model.state_dict()[key].shape}")


if __name__ == "__main__":
    main()
