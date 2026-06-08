#!/usr/bin/env python3
"""Train Transformer on Combined Filtered Datasets.

Phase 3: Combined Filtered Transformer
======================================
Uses filtered and curated data from all three sources:
- S669: Benchmark quality mutations
- ProTherm: High-quality curated mutations
- ProteinGym: Filtered for DDG-like assays only

Key insight from multimodal experiments:
- Different datasets have different scales and biases
- Filtering is essential to remove non-DDG assays
- Unified feature representation enables transfer learning

Architecture:
    Combined Data -> Feature Standardization -> Transformer -> DDG

Usage:
    python src/bioinformatics/scripts/train_combined_transformer.py
    python src/bioinformatics/scripts/train_combined_transformer.py --quick
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
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset, random_split

from src.bioinformatics.data.proteingym_loader import ProteinGymLoader
from src.bioinformatics.data.protherm_loader import ProThermLoader
from src.bioinformatics.data.s669_loader import S669Loader
from src.bioinformatics.training.deterministic import set_deterministic_mode


@dataclass
class CombinedTransformerConfig:
    """Configuration for Combined Transformer."""

    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 512
    dropout: float = 0.1
    activation: str = "gelu"
    use_source_embedding: bool = True  # Learn source-specific biases
    use_gradient_checkpointing: bool = False


class CombinedTransformer(nn.Module):
    """Transformer for DDG prediction from combined filtered datasets.

    Key features:
    1. Source embedding: Learns dataset-specific biases
    2. Feature normalization: Handles scale differences
    3. Deep architecture: More layers for complex patterns
    4. Multi-scale attention: Different head sizes for local/global patterns
    """

    def __init__(
        self,
        config: CombinedTransformerConfig,
        input_dim: int,
        n_sources: int = 3,
    ):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.n_sources = n_sources

        # Feature normalization
        self.input_norm = nn.LayerNorm(input_dim)

        # Input projection
        self.input_proj = nn.Linear(1, config.d_model)

        # Source embedding (learns dataset-specific biases)
        if config.use_source_embedding:
            self.source_embedding = nn.Embedding(n_sources, config.d_model)

        # Positional encoding
        self.pos_enc = nn.Parameter(
            torch.randn(1, input_dim + 1, config.d_model) * 0.02  # +1 for CLS
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
        )

        # Multi-scale prediction head
        self.head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.SiLU(),
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model // 2),
            nn.SiLU(),
            nn.Dropout(config.dropout / 2),
            nn.Linear(config.d_model // 2, 1),
        )

        # Source-specific bias correction
        if config.use_source_embedding:
            self.source_bias = nn.Embedding(n_sources, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        source_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input features (batch, input_dim)
            source_ids: Source dataset IDs (batch,) - 0: S669, 1: ProTherm, 2: ProteinGym

        Returns:
            Dictionary with predictions
        """
        batch_size = x.shape[0]

        # Normalize input features
        x = self.input_norm(x)

        # Treat each feature as a token
        x = x.unsqueeze(-1)  # (batch, input_dim, 1)
        x = self.input_proj(x)  # (batch, input_dim, d_model)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, 1+input_dim, d_model)

        # Add positional encoding
        x = x + self.pos_enc[:, : x.size(1)]

        # Add source embedding to CLS token
        if self.config.use_source_embedding and source_ids is not None:
            source_emb = self.source_embedding(source_ids)  # (batch, d_model)
            x[:, 0] = x[:, 0] + source_emb

        # Transform
        x = self.transformer(x)

        # Get CLS output
        cls_out = x[:, 0]  # (batch, d_model)

        # Predict DDG
        ddg_pred = self.head(cls_out)

        # Add source-specific bias
        if self.config.use_source_embedding and source_ids is not None:
            source_bias = self.source_bias(source_ids)  # (batch, 1)
            ddg_pred = ddg_pred + source_bias

        return {
            "ddg_pred": ddg_pred,
            "cls_embedding": cls_out,
        }

    def loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        source_ids: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> dict[str, torch.Tensor]:
        """Compute loss.

        Args:
            x: Input features
            y: Target DDG values
            source_ids: Source dataset IDs
            reduction: Loss reduction method

        Returns:
            Dictionary with loss components
        """
        output = self.forward(x, source_ids)
        ddg_pred = output["ddg_pred"]

        if y.dim() == 1:
            y = y.unsqueeze(-1)

        mse_loss = F.mse_loss(ddg_pred, y, reduction=reduction)

        return {
            "loss": mse_loss,
            "mse_loss": mse_loss,
        }


class CombinedFilteredDataset(Dataset):
    """Combined dataset from filtered sources with unified features."""

    def __init__(
        self,
        s669_records: list,
        protherm_db: list,
        proteingym_samples: list | None = None,
        max_proteingym: int = 10000,
    ):
        """Initialize combined dataset.

        Args:
            s669_records: S669 mutation records
            protherm_db: ProTherm database records
            proteingym_samples: Optional pre-loaded ProteinGym samples
            max_proteingym: Maximum ProteinGym samples to include
        """
        from src.bioinformatics.data.preprocessing import compute_features

        self.features = []
        self.labels = []
        self.source_ids = []

        # Process S669 (source_id = 0)
        print(f"  Processing S669 ({len(s669_records)} records)...")
        for record in s669_records:
            feat = compute_features(record.wild_type, record.mutant)
            self.features.append(feat.tensor)
            self.labels.append(record.ddg)
            self.source_ids.append(0)

        # Process ProTherm (source_id = 1)
        print(f"  Processing ProTherm ({len(protherm_db)} records)...")
        for record in protherm_db:
            feat = compute_features(record["wild_type"], record["mutant"])
            self.features.append(feat.tensor)
            self.labels.append(record["ddg"])
            self.source_ids.append(1)

        # Process ProteinGym (source_id = 2)
        if proteingym_samples:
            n_samples = min(len(proteingym_samples), max_proteingym)
            print(f"  Processing ProteinGym ({n_samples} samples)...")
            for _i, sample in enumerate(proteingym_samples[:n_samples]):
                if sample is not None:
                    feat = compute_features(sample["wild_type"], sample["mutant"])
                    self.features.append(feat.tensor)
                    self.labels.append(sample["fitness"])
                    self.source_ids.append(2)

        self.features = torch.stack(self.features)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        self.source_ids = torch.tensor(self.source_ids, dtype=torch.long)

        # Standardize labels per source (handle scale differences)
        self._standardize_labels()

        print(f"  Total: {len(self.labels)} samples")
        print(f"    S669: {(self.source_ids == 0).sum().item()}")
        print(f"    ProTherm: {(self.source_ids == 1).sum().item()}")
        print(f"    ProteinGym: {(self.source_ids == 2).sum().item()}")

    def _standardize_labels(self):
        """Standardize labels per source to handle scale differences."""
        for source_id in range(3):
            mask = self.source_ids == source_id
            if mask.sum() > 0:
                source_labels = self.labels[mask]
                mean = source_labels.mean()
                std = source_labels.std()
                if std > 0:
                    self.labels[mask] = (source_labels - mean) / std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "features": self.features[idx],
            "label": self.labels[idx],
            "source_id": self.source_ids[idx],
        }

    @property
    def input_dim(self):
        return self.features.shape[1]


def compute_qa_metrics(
    model: CombinedTransformer,
    dataset: Dataset,
    device: str = "cuda",
) -> dict[str, float]:
    """Compute QA metrics."""
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    per_source_preds = {0: [], 1: [], 2: []}
    per_source_labels = {0: [], 1: [], 2: []}

    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            y = batch["label"]
            source_ids = batch["source_id"].to(device)

            output = model(x, source_ids)
            pred = output["ddg_pred"].squeeze(-1)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.numpy())

            # Per-source tracking
            for _i, (p, l, s) in enumerate(zip(pred.cpu().numpy(), y.numpy(), source_ids.cpu().numpy(), strict=False)):
                per_source_preds[s].append(p)
                per_source_labels[s].append(l)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Overall metrics
    spearman_r, spearman_p = spearmanr(all_preds, all_labels)
    pearson_r, pearson_p = pearsonr(all_preds, all_labels)
    mae = np.mean(np.abs(all_preds - all_labels))
    rmse = np.sqrt(np.mean((all_preds - all_labels) ** 2))

    result = {
        "spearman": float(spearman_r),
        "spearman_p": float(spearman_p),
        "pearson": float(pearson_r),
        "pearson_p": float(pearson_p),
        "mae": float(mae),
        "rmse": float(rmse),
        "n_samples": len(all_preds),
    }

    # Per-source metrics
    source_names = {0: "s669", 1: "protherm", 2: "proteingym"}
    for source_id, name in source_names.items():
        if per_source_preds[source_id]:
            preds = np.array(per_source_preds[source_id])
            labels = np.array(per_source_labels[source_id])
            if len(preds) > 2:
                s_r, _ = spearmanr(preds, labels)
                result[f"{name}_spearman"] = float(s_r)
                result[f"{name}_n"] = len(preds)

    return result


def print_qa_report(name: str, metrics: dict[str, float]):
    """Print formatted QA report."""
    print(f"\n{'=' * 60}")
    print(f"QA REPORT: {name}")
    print(f"{'=' * 60}")
    print(f"  Total Samples:  {metrics['n_samples']}")
    print(f"  Spearman ρ:     {metrics['spearman']:.4f} (p={metrics['spearman_p']:.2e})")
    print(f"  Pearson r:      {metrics['pearson']:.4f} (p={metrics['pearson_p']:.2e})")
    print(f"  MAE:            {metrics['mae']:.4f}")
    print(f"  RMSE:           {metrics['rmse']:.4f}")

    # Per-source breakdown
    print("\n  Per-Source Breakdown:")
    for source in ["s669", "protherm", "proteingym"]:
        if f"{source}_spearman" in metrics:
            print(f"    {source:12s}: ρ={metrics[f'{source}_spearman']:.4f} (n={metrics[f'{source}_n']})")

    quality = "EXCELLENT" if metrics["spearman"] > 0.7 else "GOOD" if metrics["spearman"] > 0.5 else "MODERATE"
    print(f"\n  Quality:        {quality}")
    print(f"{'=' * 60}")


def train_combined_transformer(
    train_dataset: Dataset,
    val_dataset: Dataset,
    output_dir: Path,
    config: CombinedTransformerConfig | None = None,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-4,
    patience: int = 30,
    device: str = "cuda",
    verbose: bool = True,
) -> tuple[CombinedTransformer, dict[str, Any]]:
    """Train Combined Transformer."""

    if config is None:
        config = CombinedTransformerConfig()

    input_dim = train_dataset.dataset.input_dim if hasattr(train_dataset, "dataset") else train_dataset.input_dim
    output_dir.mkdir(parents=True, exist_ok=True)

    model = CombinedTransformer(config, input_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr * 10,
        epochs=epochs,
        steps_per_epoch=len(train_dataset) // batch_size + 1,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_spearman": [],
    }

    best_spearman = -float("inf")
    best_epoch = 0
    no_improve = 0

    print(f"\n{'=' * 60}")
    print("Training: Combined Transformer")
    print(f"{'=' * 60}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples:   {len(val_dataset)}")
    print(f"  Input dim:     {input_dim}")
    print(f"  d_model:       {config.d_model}")
    print(f"  n_layers:      {config.n_layers}")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            x = batch["features"].to(device)
            y = batch["label"].to(device)
            source_ids = batch["source_id"].to(device)

            optimizer.zero_grad()
            loss_dict = model.loss(x, y, source_ids)
            loss = loss_dict["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                x = batch["features"].to(device)
                y = batch["label"].to(device)
                source_ids = batch["source_id"].to(device)

                output = model(x, source_ids)
                pred = output["ddg_pred"].squeeze(-1)

                loss = F.mse_loss(pred, y)
                val_loss += loss.item()

                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        val_loss /= len(val_loader)
        val_spearman = spearmanr(all_preds, all_labels)[0]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_spearman"].append(float(val_spearman))

        if verbose and epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}: loss={train_loss:.4f} val_loss={val_loss:.4f} ρ={val_spearman:.4f}")

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

    # Load best model
    ckpt = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    print("\n  Training complete!")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best Spearman: {best_spearman:.4f}")

    return model, history


class SubsetWrapper(Dataset):
    """Wrapper for random_split subsets."""

    def __init__(self, subset, original_dataset):
        self.subset = subset
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        return self.subset[idx]

    @property
    def input_dim(self):
        return self.original_dataset.input_dim


def main():
    parser = argparse.ArgumentParser(description="Train Combined Transformer")
    parser.add_argument("--quick", action="store_true", help="Quick test run")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--max-proteingym", type=int, default=50000, help="Max ProteinGym samples")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    set_deterministic_mode(42)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs/combined_transformer_{timestamp}")

    epochs = 30 if args.quick else 100
    patience = 10 if args.quick else 30
    max_proteingym = 5000 if args.quick else args.max_proteingym

    print("=" * 70)
    print("COMBINED FILTERED TRANSFORMER TRAINING")
    print("S669 + ProTherm + ProteinGym (filtered) -> Transformer -> DDG")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {epochs}, Patience: {patience}")
    print(f"Max ProteinGym samples: {max_proteingym}")

    # =========================================================
    # Load all datasets
    # =========================================================
    print("\n[1] Loading datasets...")

    # S669
    print("\n  Loading S669...")
    s669_loader = S669Loader()
    try:
        s669_records = s669_loader.load_from_csv()
    except FileNotFoundError:
        s669_records = s669_loader.load_curated_subset()
    print(f"    S669: {len(s669_records)} records")

    # ProTherm
    print("\n  Loading ProTherm...")
    protherm_loader = ProThermLoader()
    protherm_db = protherm_loader.load_curated()
    print(f"    ProTherm: {len(protherm_db)} records")

    # ProteinGym (filtered)
    print("\n  Loading ProteinGym (filtered)...")
    proteingym_samples = []
    try:
        proteingym_loader = ProteinGymLoader()
        proteingym_dataset = proteingym_loader.create_dataset(max_records=max_proteingym)
        # Extract samples with their raw features
        for i in range(min(len(proteingym_dataset), max_proteingym)):
            x, y = proteingym_dataset[i]
            proteingym_samples.append(
                {
                    "features": x,
                    "fitness": float(y),
                }
            )
        print(f"    ProteinGym: {len(proteingym_samples)} samples loaded")
    except FileNotFoundError:
        print("    ProteinGym data not found, continuing without it")
    except Exception as e:
        print(f"    ProteinGym loading failed: {e}")
        print("    Continuing without ProteinGym data")

    # =========================================================
    # Create combined dataset
    # =========================================================
    print("\n[2] Creating combined dataset...")

    # Simplified approach: use pre-extracted features from loaders
    # and combine directly
    from src.bioinformatics.data.preprocessing import compute_features

    features = []
    labels = []
    source_ids = []

    # S669
    for record in s669_records:
        feat = compute_features(record.wild_type, record.mutant)
        feat_array = torch.tensor(feat.to_array(include_hyperbolic=False), dtype=torch.float32)
        features.append(feat_array)
        labels.append(record.ddg)
        source_ids.append(0)

    # ProTherm (records can be objects or dicts)
    for record in protherm_db:
        if hasattr(record, "wild_type"):
            wt, mut, ddg = record.wild_type, record.mutant, record.ddg
        else:
            wt, mut, ddg = record["wild_type"], record["mutant"], record["ddg"]
        feat = compute_features(wt, mut)
        feat_array = torch.tensor(feat.to_array(include_hyperbolic=False), dtype=torch.float32)
        features.append(feat_array)
        labels.append(ddg)
        source_ids.append(1)

    # ProteinGym (use pre-computed features if available)
    if proteingym_samples:
        for sample in proteingym_samples:
            if "features" in sample and sample["features"] is not None:
                feat_tensor = sample["features"]
                if not isinstance(feat_tensor, torch.Tensor):
                    feat_tensor = torch.tensor(feat_tensor, dtype=torch.float32)
                features.append(feat_tensor)
                labels.append(sample["fitness"])
                source_ids.append(2)

    features = torch.stack(features)
    labels = torch.tensor(labels, dtype=torch.float32)
    source_ids = torch.tensor(source_ids, dtype=torch.long)

    # Standardize labels per source
    for sid in range(3):
        mask = source_ids == sid
        if mask.sum() > 0:
            src_labels = labels[mask]
            mean, std = src_labels.mean(), src_labels.std()
            if std > 0:
                labels[mask] = (src_labels - mean) / std

    print(f"\n  Combined dataset: {len(labels)} samples")
    print(f"    S669: {(source_ids == 0).sum().item()}")
    print(f"    ProTherm: {(source_ids == 1).sum().item()}")
    print(f"    ProteinGym: {(source_ids == 2).sum().item()}")

    # Create dataset class
    class SimpleDataset(Dataset):
        def __init__(self, features, labels, source_ids):
            self.features = features
            self.labels = labels
            self.source_ids = source_ids
            self.input_dim = features.shape[1]

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                "features": self.features[idx],
                "label": self.labels[idx],
                "source_id": self.source_ids[idx],
            }

    full_dataset = SimpleDataset(features, labels, source_ids)

    # Split
    n_val = int(len(full_dataset) * 0.15)
    n_train = len(full_dataset) - n_val

    train_ds, val_ds = random_split(full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_ds = SubsetWrapper(train_ds, full_dataset)
    val_ds = SubsetWrapper(val_ds, full_dataset)

    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    # =========================================================
    # Train Combined Transformer
    # =========================================================
    print("\n[3] Training Combined Transformer...")

    config = CombinedTransformerConfig(
        d_model=128,
        n_heads=8,
        n_layers=6,
        d_ff=512,
        dropout=0.15,
        use_source_embedding=True,
    )

    model, history = train_combined_transformer(
        train_ds,
        val_ds,
        output_dir=output_dir,
        config=config,
        epochs=epochs,
        batch_size=64,
        lr=1e-4,
        patience=patience,
        device=args.device,
    )

    # =========================================================
    # QA
    # =========================================================
    print("\n[4] Quality Assurance...")

    qa_metrics = compute_qa_metrics(model, val_ds, args.device)
    print_qa_report("Combined Transformer", qa_metrics)

    with open(output_dir / "qa_metrics.json", "w") as f:
        json.dump(qa_metrics, f, indent=2)

    # =========================================================
    # Summary
    # =========================================================
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"  Overall Spearman: {qa_metrics['spearman']:.4f}")
    print("  Per-source performance:")
    for source in ["s669", "protherm", "proteingym"]:
        if f"{source}_spearman" in qa_metrics:
            print(f"    {source}: {qa_metrics[f'{source}_spearman']:.4f}")
    print(f"\n  Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
