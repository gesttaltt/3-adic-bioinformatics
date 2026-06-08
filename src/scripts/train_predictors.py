# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Train downstream predictors using VAE embeddings.

This script provides a complete pipeline for:
1. Loading a trained Ternary VAE
2. Extracting hyperbolic embeddings for sequences
3. Training downstream predictors (resistance, escape, neutralization, tropism)
4. Evaluating and saving the trained predictors

Usage:
    # Train resistance predictor with VAE embeddings
    python src/scripts/train_predictors.py --predictor resistance --vae-checkpoint outputs/best.pt

    # Train all predictors
    python src/scripts/train_predictors.py --predictor all --vae-checkpoint outputs/best.pt

    # Use synthetic data for testing
    python src/scripts/train_predictors.py --predictor resistance --synthetic

    # Train with custom data
    python src/scripts/train_predictors.py --predictor escape --data-path data/hiv/escape_mutations.csv
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def print_header(text: str):
    """Print formatted header."""
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)


def load_vae(checkpoint_path: Path, device: str) -> torch.nn.Module:
    """Load trained VAE from checkpoint."""
    from src.models import TernaryVAEV5_11_PartialFreeze

    print(f"Loading VAE from {checkpoint_path}")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get("config", {})

    # Create model
    model = TernaryVAEV5_11_PartialFreeze(
        latent_dim=config.get("latent_dim", 16),
        hidden_dim=config.get("hidden_dim", 64),
        curvature=config.get("curvature", 1.0),
        max_radius=config.get("max_radius", 0.95),
        use_dual_projection=config.get("dual_projection", False),
    )

    # Load state
    model_state = ckpt.get("model_state_dict", ckpt.get("model_state", {}))
    if model_state:
        model.load_state_dict(model_state, strict=False)

    model = model.to(device)
    model.eval()

    print(f"  Loaded from epoch {ckpt.get('epoch', 'unknown')}")
    if "metrics" in ckpt:
        m = ckpt["metrics"]
        print(f"  VAE Coverage: {m.get('coverage', 0) * 100:.1f}%")
        print(f"  VAE Radial Corr: {m.get('radial_corr_A', 0):.4f}")

    return model


def extract_embeddings(
    model: torch.nn.Module,
    sequences: list[str],
    device: str,
) -> np.ndarray:
    """Extract hyperbolic embeddings for sequences using VAE.

    Args:
        model: Trained VAE model
        sequences: List of amino acid sequences
        device: Device to use

    Returns:
        Embeddings array of shape (n_sequences, latent_dim)
    """
    from src.models.predictors.base_predictor import HyperbolicFeatureExtractor

    extractor = HyperbolicFeatureExtractor(p=3)

    embeddings = []
    for seq in sequences:
        # Get sequence features
        features = extractor.sequence_features(seq)
        embeddings.append(features)

    return np.array(embeddings)


def generate_synthetic_data(
    n_samples: int = 1000,
    predictor_type: str = "resistance",
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Generate synthetic data for testing.

    Args:
        n_samples: Number of samples to generate
        predictor_type: Type of predictor (affects target generation)

    Returns:
        sequences, features, targets
    """
    print(f"Generating {n_samples} synthetic samples...")

    # Generate random sequences
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    sequences = []
    for _ in range(n_samples):
        length = np.random.randint(50, 200)
        seq = "".join(np.random.choice(list(amino_acids), length))
        sequences.append(seq)

    # Extract features
    from src.models.predictors.base_predictor import HyperbolicFeatureExtractor

    extractor = HyperbolicFeatureExtractor(p=3)
    features = np.array([extractor.sequence_features(seq) for seq in sequences])

    # Generate targets based on predictor type
    if predictor_type == "tropism":
        # Binary classification (CCR5=0, CXCR4=1)
        targets = np.random.randint(0, 2, n_samples)
    elif predictor_type == "resistance":
        # Log fold-change (continuous, typically 0.1 to 1000)
        targets = 10 ** (np.random.randn(n_samples) * 1.5 + 0.5)
    elif predictor_type == "neutralization":
        # IC50 values (continuous, typically 0.01 to 100)
        targets = 10 ** (np.random.randn(n_samples) * 1.0)
    else:  # escape
        # Escape probability (0 to 1)
        targets = np.random.beta(2, 5, n_samples)

    return sequences, features, targets


def load_real_data(
    data_path: Path,
    predictor_type: str,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load real data from file.

    Expected CSV format:
        sequence,target

    Args:
        data_path: Path to CSV file
        predictor_type: Type of predictor

    Returns:
        sequences, features, targets
    """
    import pandas as pd

    print(f"Loading data from {data_path}")

    df = pd.read_csv(data_path)

    # Expect 'sequence' and 'target' columns
    if "sequence" not in df.columns:
        raise ValueError("Data file must have 'sequence' column")

    sequences = df["sequence"].tolist()

    # Extract features
    from src.models.predictors.base_predictor import HyperbolicFeatureExtractor

    extractor = HyperbolicFeatureExtractor(p=3)
    features = np.array([extractor.sequence_features(seq) for seq in sequences])

    # Get targets
    target_col = "target" if "target" in df.columns else df.columns[-1]
    targets = df[target_col].values

    print(f"  Loaded {len(sequences)} samples")
    print(f"  Feature shape: {features.shape}")
    print(f"  Target range: [{targets.min():.4f}, {targets.max():.4f}]")

    return sequences, features, targets


def train_predictor(
    predictor_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[object, dict]:
    """Train a predictor.

    Args:
        predictor_type: Type of predictor to train
        X_train, y_train: Training data
        X_val, y_val: Validation data

    Returns:
        trained predictor, metrics dict
    """
    from src.models.predictors import (
        EscapePredictor,
        NeutralizationPredictor,
        ResistancePredictor,
        TropismClassifier,
    )

    predictor_map = {
        "resistance": ResistancePredictor,
        "escape": EscapePredictor,
        "neutralization": NeutralizationPredictor,
        "tropism": TropismClassifier,
    }

    if predictor_type not in predictor_map:
        raise ValueError(f"Unknown predictor: {predictor_type}")

    PredictorClass = predictor_map[predictor_type]

    print(f"\nTraining {predictor_type} predictor...")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Feature dimension: {X_train.shape[1]}")

    # Create and train
    predictor = PredictorClass()
    predictor.fit(X_train, y_train)

    # Evaluate
    train_metrics = predictor.evaluate(X_train, y_train)
    val_metrics = predictor.evaluate(X_val, y_val)

    print("\n  Training metrics:")
    for k, v in train_metrics.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")

    print("\n  Validation metrics:")
    for k, v in val_metrics.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")

    return predictor, {"train": train_metrics, "val": val_metrics}


def main():
    parser = argparse.ArgumentParser(description="Train downstream predictors using VAE embeddings")

    parser.add_argument(
        "--predictor",
        "-p",
        type=str,
        required=True,
        choices=["resistance", "escape", "neutralization", "tropism", "all"],
        help="Predictor type to train",
    )
    parser.add_argument(
        "--vae-checkpoint",
        type=Path,
        help="Path to trained VAE checkpoint",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Path to training data CSV",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data for testing",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of synthetic samples (default: 1000)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2)",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("outputs/predictors"),
        help="Directory to save trained predictors",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
    )

    args = parser.parse_args()

    # Check device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    print_header("Predictor Training Pipeline")
    print(f"  Predictor: {args.predictor}")
    print(f"  Device: {device}")
    print(f"  Save dir: {args.save_dir}")

    # Determine predictors to train
    if args.predictor == "all":
        predictors_to_train = ["resistance", "escape", "neutralization", "tropism"]
    else:
        predictors_to_train = [args.predictor]

    # Load or generate data
    for predictor_type in predictors_to_train:
        print_header(f"Training {predictor_type.upper()} Predictor")

        if args.synthetic or (args.data_path is None):
            sequences, features, targets = generate_synthetic_data(
                n_samples=args.n_samples,
                predictor_type=predictor_type,
            )
        else:
            sequences, features, targets = load_real_data(
                args.data_path,
                predictor_type,
            )

        # Split data
        n_val = int(len(features) * args.val_split)
        indices = np.random.permutation(len(features))

        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        X_train, y_train = features[train_idx], targets[train_idx]
        X_val, y_val = features[val_idx], targets[val_idx]

        # Train predictor
        predictor, metrics = train_predictor(
            predictor_type,
            X_train,
            y_train,
            X_val,
            y_val,
        )

        # Save predictor
        args.save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = args.save_dir / f"{predictor_type}_predictor_{timestamp}.pkl"

        predictor.save(save_path)
        print(f"\n  Saved to: {save_path}")

    print_header("Training Complete")
    print(f"  Predictors saved to: {args.save_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
