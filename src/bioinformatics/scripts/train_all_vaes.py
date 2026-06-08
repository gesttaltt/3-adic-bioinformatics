#!/usr/bin/env python3
"""Train all three specialist DDG VAEs.

Usage:
    python src/bioinformatics/scripts/train_all_vaes.py [--quick]

    --quick: Run a quick test with reduced epochs
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[3]))

import argparse
from datetime import datetime

import torch

from src.bioinformatics.data.proteingym_loader import ProteinGymLoader
from src.bioinformatics.data.protherm_loader import ProThermLoader
from src.bioinformatics.data.s669_loader import S669Loader
from src.bioinformatics.training.deterministic import DeterministicConfig
from src.bioinformatics.training.train_ddg_vae import (
    TrainingConfig,
    train_vae_protherm,
    train_vae_s669,
    train_vae_wide,
)


def main():
    parser = argparse.ArgumentParser(description="Train all DDG VAEs")
    parser.add_argument("--quick", action="store_true", help="Quick test run")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--skip-wide", action="store_true", help="Skip VAE-Wide (large)")
    args = parser.parse_args()

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output = Path(f"outputs/ddg_vae_training_{timestamp}")
    base_output.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DDG VAE Training Pipeline")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Output: {base_output}")
    print(f"Quick mode: {args.quick}")
    print()

    # Quick mode config
    if args.quick:
        quick_config = TrainingConfig(
            epochs=10,
            batch_size=16,
            early_stopping_patience=5,
            log_every=1,
            deterministic=DeterministicConfig(seed=42),
        )
    else:
        quick_config = None

    # ========================================
    # 1. Train VAE-S669
    # ========================================
    print("\n" + "=" * 60)
    print("[1/3] Training VAE-S669 (Benchmark Specialist)")
    print("=" * 60)

    s669_loader = S669Loader()
    s669_records = s669_loader.load_curated_subset()
    s669_dataset = s669_loader.create_dataset(s669_records)
    print(f"S669 dataset: {len(s669_dataset)} samples, {s669_dataset.feature_dim} features")

    train_vae_s669(
        dataset=s669_dataset,
        output_dir=base_output / "vae_s669",
        config=quick_config,
        use_hyperbolic=False,  # No hyperbolic until we have AA embeddings
        device=args.device,
    )
    print("VAE-S669 training complete!")

    # ========================================
    # 2. Train VAE-ProTherm
    # ========================================
    print("\n" + "=" * 60)
    print("[2/3] Training VAE-ProTherm (High-Quality Specialist)")
    print("=" * 60)

    protherm_loader = ProThermLoader()
    protherm_db = protherm_loader.load_curated()
    protherm_dataset = protherm_loader.create_dataset(protherm_db)
    print(f"ProTherm dataset: {len(protherm_dataset)} samples, {protherm_dataset.feature_dim} features")

    # ProTherm has more features (includes structure info)
    protherm_config = quick_config
    if protherm_config is None:
        protherm_config = TrainingConfig(
            epochs=200,
            batch_size=16,
            learning_rate=5e-5,
            early_stopping_patience=30,
            deterministic=DeterministicConfig(seed=42),
        )

    train_vae_protherm(
        dataset=protherm_dataset,
        output_dir=base_output / "vae_protherm",
        config=protherm_config,
        use_hyperbolic=False,
        device=args.device,
    )
    print("VAE-ProTherm training complete!")

    # ========================================
    # 3. Train VAE-Wide (optional)
    # ========================================
    if not args.skip_wide:
        print("\n" + "=" * 60)
        print("[3/3] Training VAE-Wide (Diversity Specialist)")
        print("=" * 60)

        pg_loader = ProteinGymLoader(data_dir=Path("data/bioinformatics/ddg/proteingym/DMS_ProteinGym_substitutions"))

        # Load subset for training (full dataset is too large for memory)
        max_records = 10000 if args.quick else 100000
        print(f"Loading up to {max_records:,} ProteinGym records...")
        pg_dataset = pg_loader.create_dataset(
            max_records=max_records,
            use_fitness_as_label=True,
        )
        print(f"ProteinGym dataset: {len(pg_dataset)} samples, {pg_dataset.feature_dim} features")

        wide_config = quick_config
        if wide_config is None:
            wide_config = TrainingConfig(
                epochs=50,
                batch_size=128,
                learning_rate=1e-3,
                early_stopping_patience=10,
                deterministic=DeterministicConfig(seed=42),
            )

        train_vae_wide(
            dataset=pg_dataset,
            output_dir=base_output / "vae_wide",
            config=wide_config,
            use_hyperbolic=False,
            device=args.device,
        )
        print("VAE-Wide training complete!")
    else:
        print("\n[3/3] Skipping VAE-Wide (--skip-wide flag)")

    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nOutputs saved to: {base_output}")
    print("\nTrained models:")
    print(f"  - VAE-S669:    {base_output / 'vae_s669' / 'best.pt'}")
    print(f"  - VAE-ProTherm: {base_output / 'vae_protherm' / 'best.pt'}")
    if not args.skip_wide:
        print(f"  - VAE-Wide:    {base_output / 'vae_wide' / 'best.pt'}")


if __name__ == "__main__":
    main()
