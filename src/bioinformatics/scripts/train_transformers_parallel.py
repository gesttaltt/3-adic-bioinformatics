#!/usr/bin/env python3
"""Train Three Parallel Transformers for DDG Prediction.

Phase 1: Three Dataset-Specific Transformers (Sequential with QA)
================================================================
Each transformer is trained on its specific dataset:
1. Transformer-S669: Trained on S669 benchmark
2. Transformer-ProTherm: Trained on curated ProTherm
3. Transformer-Wide: Trained on filtered ProteinGym

Each training includes:
- Full training loop with early stopping
- Quality Assurance metrics after training
- Checkpoint saving with validation metrics

Usage:
    python src/bioinformatics/scripts/train_transformers_parallel.py
    python src/bioinformatics/scripts/train_transformers_parallel.py --quick
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split

from src.bioinformatics.data.proteingym_loader import ProteinGymLoader
from src.bioinformatics.data.protherm_loader import ProThermLoader
from src.bioinformatics.data.s669_loader import S669Loader
from src.bioinformatics.training.deterministic import set_deterministic_mode


@dataclass
class TransformerConfig:
    """Configuration for DDG Transformer."""

    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 256
    dropout: float = 0.1
    max_seq_len: int = 64  # Input feature dimension
    activation: str = "gelu"


class DDGTransformer(nn.Module):
    """Transformer for DDG prediction from raw features.

    Unlike the embedding transformer (which takes VAE latents),
    this transformer processes raw mutation features directly,
    treating feature dimensions as a sequence.
    """

    def __init__(self, config: TransformerConfig, input_dim: int):
        super().__init__()
        self.config = config
        self.input_dim = input_dim

        # Input projection: each feature becomes a token
        self.input_proj = nn.Linear(1, config.d_model)

        # Positional encoding for feature dimensions
        self.pos_enc = nn.Parameter(torch.randn(1, input_dim, config.d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
            norm_first=True,  # Pre-LayerNorm for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
        )

        # CLS token for global representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)

        # Prediction head with residual structure
        self.head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.SiLU(),
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model // 2),
            nn.SiLU(),
            nn.Linear(config.d_model // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input features (batch, input_dim)

        Returns:
            Dictionary with 'ddg_pred' and 'attention_weights'
        """
        batch_size = x.shape[0]

        # Treat each feature as a token: (batch, input_dim) -> (batch, input_dim, 1)
        x = x.unsqueeze(-1)

        # Project to model dimension
        x = self.input_proj(x)  # (batch, input_dim, d_model)

        # Add positional encoding
        x = x + self.pos_enc

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, 1+input_dim, d_model)

        # Transform
        x = self.transformer(x)

        # Use CLS token for prediction
        cls_out = x[:, 0]  # (batch, d_model)

        # Predict DDG
        ddg_pred = self.head(cls_out)

        return {
            "ddg_pred": ddg_pred,
            "cls_embedding": cls_out,
        }

    def loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        reduction: str = "mean",
    ) -> dict[str, torch.Tensor]:
        """Compute MSE loss.

        Args:
            x: Input features
            y: Target DDG values
            reduction: Loss reduction method

        Returns:
            Dictionary with loss components
        """
        output = self.forward(x)
        ddg_pred = output["ddg_pred"]

        # Ensure y has correct shape
        if y.dim() == 1:
            y = y.unsqueeze(-1)

        mse_loss = F.mse_loss(ddg_pred, y, reduction=reduction)

        return {
            "loss": mse_loss,
            "mse_loss": mse_loss,
        }


def compute_qa_metrics(
    model: nn.Module,
    dataset: Dataset,
    device: str = "cuda",
) -> dict[str, float]:
    """Compute comprehensive QA metrics for a trained model.

    Args:
        model: Trained transformer model
        dataset: Dataset to evaluate on
        device: Computation device

    Returns:
        Dictionary of QA metrics
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    all_losses = []

    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (tuple, list)):
                x, y = batch
            else:
                x, y = batch["features"], batch["label"]

            x = x.to(device)
            y = y.to(device)

            output = model(x)
            ddg_pred = output["ddg_pred"].squeeze(-1)

            loss = F.mse_loss(ddg_pred, y, reduction="none")

            all_preds.extend(ddg_pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_losses.extend(loss.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_losses = np.array(all_losses)

    # Compute metrics
    spearman_r, spearman_p = spearmanr(all_preds, all_labels)
    pearson_r, pearson_p = pearsonr(all_preds, all_labels)
    mae = np.mean(np.abs(all_preds - all_labels))
    rmse = np.sqrt(np.mean((all_preds - all_labels) ** 2))

    # Error analysis
    residuals = all_preds - all_labels

    return {
        "spearman": float(spearman_r),
        "spearman_p": float(spearman_p),
        "pearson": float(pearson_r),
        "pearson_p": float(pearson_p),
        "mae": float(mae),
        "rmse": float(rmse),
        "mean_loss": float(np.mean(all_losses)),
        "std_loss": float(np.std(all_losses)),
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
        "n_samples": len(all_preds),
    }


def print_qa_report(name: str, metrics: dict[str, float]):
    """Print formatted QA report."""
    print(f"\n{'=' * 60}")
    print(f"QA REPORT: {name}")
    print(f"{'=' * 60}")
    print(f"  Samples:        {metrics['n_samples']}")
    print(f"  Spearman ρ:     {metrics['spearman']:.4f} (p={metrics['spearman_p']:.2e})")
    print(f"  Pearson r:      {metrics['pearson']:.4f} (p={metrics['pearson_p']:.2e})")
    print(f"  MAE:            {metrics['mae']:.4f}")
    print(f"  RMSE:           {metrics['rmse']:.4f}")
    print(f"  Residual mean:  {metrics['residual_mean']:.4f}")
    print(f"  Residual std:   {metrics['residual_std']:.4f}")

    # Quality assessment
    quality = (
        "EXCELLENT"
        if metrics["spearman"] > 0.7
        else "GOOD"
        if metrics["spearman"] > 0.5
        else "MODERATE"
        if metrics["spearman"] > 0.3
        else "NEEDS IMPROVEMENT"
    )
    print(f"\n  Quality:        {quality}")
    print(f"{'=' * 60}")


def train_transformer(
    name: str,
    train_dataset: Dataset,
    val_dataset: Dataset,
    input_dim: int,
    output_dir: Path,
    config: TransformerConfig | None = None,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-4,
    patience: int = 20,
    device: str = "cuda",
    verbose: bool = True,
) -> tuple[DDGTransformer, dict[str, Any]]:
    """Train a DDG Transformer with early stopping and QA.

    Args:
        name: Model name for logging
        train_dataset: Training dataset
        val_dataset: Validation dataset
        input_dim: Input feature dimension
        output_dir: Output directory
        config: Transformer configuration
        epochs: Maximum epochs
        batch_size: Batch size
        lr: Learning rate
        patience: Early stopping patience
        device: Computation device
        verbose: Print progress

    Returns:
        Trained model and training history
    """
    if config is None:
        config = TransformerConfig(d_model=64, n_heads=4, n_layers=3)

    output_dir.mkdir(parents=True, exist_ok=True)

    model = DDGTransformer(config, input_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_spearman": [],
        "val_pearson": [],
    }

    best_spearman = -float("inf")
    best_epoch = 0
    no_improve = 0

    print(f"\n{'=' * 60}")
    print(f"Training: {name}")
    print(f"{'=' * 60}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples:   {len(val_dataset)}")
    print(f"  Input dim:     {input_dim}")
    print(f"  d_model:       {config.d_model}")
    print(f"  n_layers:      {config.n_layers}")
    print(f"  n_heads:       {config.n_heads}")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            if isinstance(batch, (tuple, list)):
                x, y = batch
            else:
                x, y = batch["features"], batch["label"]

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            loss_dict = model.loss(x, y)
            loss = loss_dict["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (tuple, list)):
                    x, y = batch
                else:
                    x, y = batch["features"], batch["label"]

                x = x.to(device)
                y = y.to(device)

                output = model(x)
                ddg_pred = output["ddg_pred"].squeeze(-1)

                loss = F.mse_loss(ddg_pred, y)
                val_loss += loss.item()

                all_preds.extend(ddg_pred.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        val_loss /= len(val_loader)
        val_spearman = spearmanr(all_preds, all_labels)[0]
        val_pearson = pearsonr(all_preds, all_labels)[0]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_spearman"].append(float(val_spearman))
        history["val_pearson"].append(float(val_pearson))

        if verbose and epoch % 10 == 0:
            print(
                f"  Epoch {epoch:3d}: loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"ρ={val_spearman:.4f} r={val_pearson:.4f}"
            )

        # Early stopping
        if val_spearman > best_spearman:
            best_spearman = val_spearman
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "spearman": val_spearman,
                    "pearson": val_pearson,
                },
                output_dir / "best.pt",
            )
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} (best: {best_epoch})")
                break

    # Save final
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "history": history,
        },
        output_dir / "final.pt",
    )

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Load best model for QA
    ckpt = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    print("\n  Training complete!")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best Spearman: {best_spearman:.4f}")

    return model, history


class FeatureDataset(Dataset):
    """Simple dataset wrapper for (features, labels)."""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_s669_data(device: str = "cuda") -> tuple[Dataset, Dataset, int]:
    """Load and prepare S669 dataset."""
    print("\n[Loading S669 dataset...]")

    loader = S669Loader()
    try:
        records = loader.load_from_csv()
    except FileNotFoundError:
        # Fall back to curated subset
        print("  Full S669 not found, using curated subset")
        records = loader.load_curated_subset()

    # Create dataset (handles feature extraction internally)
    full_dataset = loader.create_dataset(records=records)

    # Extract features and labels from the dataset
    features = []
    labels = []

    for i in range(len(full_dataset)):
        x, y = full_dataset[i]
        features.append(x)
        labels.append(y)

    features = torch.stack(features)
    labels = torch.tensor(labels, dtype=torch.float32)

    print(f"  Loaded {len(labels)} mutations, feature dim={features.shape[1]}")

    # Split
    dataset = FeatureDataset(features, labels)
    n_val = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val

    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    return train_ds, val_ds, features.shape[1]


def load_protherm_data(device: str = "cuda") -> tuple[Dataset, Dataset, int]:
    """Load and prepare ProTherm dataset."""
    print("\n[Loading ProTherm dataset...]")

    loader = ProThermLoader()
    db = loader.load_curated()
    base_dataset = loader.create_dataset(db)

    # Extract features and labels
    features = []
    labels = []

    for i in range(len(base_dataset)):
        x, y = base_dataset[i]
        features.append(x)
        labels.append(y)

    features = torch.stack(features)
    labels = torch.tensor(labels, dtype=torch.float32)

    print(f"  Loaded {len(labels)} mutations, feature dim={features.shape[1]}")

    # Split
    dataset = FeatureDataset(features, labels)
    n_val = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val

    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    return train_ds, val_ds, features.shape[1]


def load_proteingym_data(
    max_samples: int = 100000,
    device: str = "cuda",
) -> tuple[Dataset, Dataset, int]:
    """Load and prepare filtered ProteinGym dataset."""
    print("\n[Loading ProteinGym dataset (filtered)...]")

    loader = ProteinGymLoader()
    dataset = loader.create_filtered_dataset(
        max_samples=max_samples,
        exclude_extreme_scores=True,
    )

    # Extract features and labels
    features = []
    labels = []

    for i in range(len(dataset)):
        x, y = dataset[i]
        features.append(x)
        labels.append(y)

    features = torch.stack(features)
    labels = torch.tensor(labels, dtype=torch.float32)

    print(f"  Loaded {len(labels)} mutations, feature dim={features.shape[1]}")

    # Split
    full_dataset = FeatureDataset(features, labels)
    n_val = int(len(full_dataset) * 0.1)  # Smaller val split for large dataset
    n_train = len(full_dataset) - n_val

    train_ds, val_ds = random_split(full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    return train_ds, val_ds, features.shape[1]


def main():
    parser = argparse.ArgumentParser(description="Train Three Parallel Transformers")
    parser.add_argument("--quick", action="store_true", help="Quick test run")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--skip-s669", action="store_true", help="Skip S669 transformer")
    parser.add_argument("--skip-protherm", action="store_true", help="Skip ProTherm transformer")
    parser.add_argument("--skip-wide", action="store_true", help="Skip Wide transformer")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    set_deterministic_mode(42)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output = Path(f"outputs/transformers_parallel_{timestamp}")

    epochs = 30 if args.quick else 150
    patience = 10 if args.quick else 30
    max_wide_samples = 10000 if args.quick else 100000

    print("=" * 70)
    print("THREE PARALLEL TRANSFORMERS TRAINING")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Output: {base_output}")
    print(f"Epochs: {epochs}, Patience: {patience}")

    results = {}

    # =========================================================
    # TRANSFORMER 1: S669
    # =========================================================
    if not args.skip_s669:
        print("\n" + "=" * 70)
        print("TRANSFORMER 1: S669 (Benchmark Dataset)")
        print("=" * 70)

        train_ds, val_ds, input_dim = load_s669_data(args.device)

        config = TransformerConfig(
            d_model=64,
            n_heads=4,
            n_layers=3,
            d_ff=256,
            dropout=0.1,
        )

        model_s669, history_s669 = train_transformer(
            name="Transformer-S669",
            train_dataset=train_ds,
            val_dataset=val_ds,
            input_dim=input_dim,
            output_dir=base_output / "transformer_s669",
            config=config,
            epochs=epochs,
            batch_size=32,
            lr=1e-4,
            patience=patience,
            device=args.device,
        )

        # QA
        qa_metrics = compute_qa_metrics(model_s669, val_ds, args.device)
        print_qa_report("Transformer-S669", qa_metrics)

        with open(base_output / "transformer_s669" / "qa_metrics.json", "w") as f:
            json.dump(qa_metrics, f, indent=2)

        results["s669"] = {
            "best_spearman": max(history_s669["val_spearman"]),
            "qa_metrics": qa_metrics,
        }

    # =========================================================
    # TRANSFORMER 2: ProTherm
    # =========================================================
    if not args.skip_protherm:
        print("\n" + "=" * 70)
        print("TRANSFORMER 2: ProTherm (High-Quality Curated)")
        print("=" * 70)

        train_ds, val_ds, input_dim = load_protherm_data(args.device)

        config = TransformerConfig(
            d_model=96,  # Larger for richer data
            n_heads=6,
            n_layers=4,
            d_ff=384,
            dropout=0.05,  # Less dropout for cleaner data
        )

        model_protherm, history_protherm = train_transformer(
            name="Transformer-ProTherm",
            train_dataset=train_ds,
            val_dataset=val_ds,
            input_dim=input_dim,
            output_dir=base_output / "transformer_protherm",
            config=config,
            epochs=epochs,
            batch_size=16,
            lr=5e-5,
            patience=patience,
            device=args.device,
        )

        # QA
        qa_metrics = compute_qa_metrics(model_protherm, val_ds, args.device)
        print_qa_report("Transformer-ProTherm", qa_metrics)

        with open(base_output / "transformer_protherm" / "qa_metrics.json", "w") as f:
            json.dump(qa_metrics, f, indent=2)

        results["protherm"] = {
            "best_spearman": max(history_protherm["val_spearman"]),
            "qa_metrics": qa_metrics,
        }

    # =========================================================
    # TRANSFORMER 3: Wide (ProteinGym Filtered)
    # =========================================================
    if not args.skip_wide:
        print("\n" + "=" * 70)
        print("TRANSFORMER 3: Wide (ProteinGym Filtered)")
        print("=" * 70)

        train_ds, val_ds, input_dim = load_proteingym_data(max_wide_samples, args.device)

        config = TransformerConfig(
            d_model=128,  # Larger for diverse data
            n_heads=8,
            n_layers=4,
            d_ff=512,
            dropout=0.15,  # More dropout for noisy data
        )

        model_wide, history_wide = train_transformer(
            name="Transformer-Wide",
            train_dataset=train_ds,
            val_dataset=val_ds,
            input_dim=input_dim,
            output_dir=base_output / "transformer_wide",
            config=config,
            epochs=epochs,
            batch_size=128,
            lr=1e-3,
            patience=patience // 2,  # Less patience for large dataset
            device=args.device,
        )

        # QA
        qa_metrics = compute_qa_metrics(model_wide, val_ds, args.device)
        print_qa_report("Transformer-Wide", qa_metrics)

        with open(base_output / "transformer_wide" / "qa_metrics.json", "w") as f:
            json.dump(qa_metrics, f, indent=2)

        results["wide"] = {
            "best_spearman": max(history_wide["val_spearman"]),
            "qa_metrics": qa_metrics,
        }

    # =========================================================
    # SUMMARY
    # =========================================================
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 70)

    for name, res in results.items():
        print(f"\n{name.upper()}:")
        print(f"  Best Spearman: {res['best_spearman']:.4f}")
        print(f"  QA Spearman:   {res['qa_metrics']['spearman']:.4f}")
        print(f"  QA MAE:        {res['qa_metrics']['mae']:.4f}")

    # Save summary
    with open(base_output / "summary.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\n\nResults saved to: {base_output}")
    print("\nNext steps:")
    print("  1. Review QA metrics for each transformer")
    print("  2. Run train_stochastic_transformer.py for VAE+Refiner embedding transformer")
    print("  3. Run train_combined_transformer.py for combined filtered dataset")


if __name__ == "__main__":
    main()
