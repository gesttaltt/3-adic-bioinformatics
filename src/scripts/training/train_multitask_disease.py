#!/usr/bin/env python3
# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Multi-Task Disease Predictor Training.

Implements GradNorm-based multi-task learning for predicting across
multiple disease domains simultaneously.

Features:
- Multi-head architecture for disease-specific predictions
- GradNorm for automatic task weight balancing
- Shared encoder with task-specific decoders
- Cross-disease transfer learning

NEW (v2.0): Enhanced with SOTA components:
- VariantEscapeHead: EVEscape-inspired escape prediction (fitness, immune, drug resistance)
- ProteinGymEvaluator: Standardized quality, novelty, diversity metrics

Prediction Tasks:
- Resistance prediction (HIV, Cancer)
- Escape prediction (HIV, Flu)
- Binding affinity (All)
- Immunogenicity (All)
- Structure stability (All)
- Codon optimization (All)

Hardware: RTX 2060 SUPER (8GB VRAM)
Estimated Duration: 3-5 hours

Usage:
    # Train multi-task predictor
    python src/scripts/training/train_multitask_disease.py

    # Specific diseases
    python src/scripts/training/train_multitask_disease.py --diseases hiv cancer

    # Specific tasks
    python src/scripts/training/train_multitask_disease.py --tasks resistance escape

    # With EVEscape-inspired variant escape head (NEW)
    python src/scripts/training/train_multitask_disease.py --use-escape-head

    # With ProteinGym evaluation (NEW)
    python src/scripts/training/train_multitask_disease.py --evaluate
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Import new SOTA components
try:
    from src.diseases import VariantEscapeHead  # noqa: F401
    from src.evaluation import ProteinGymEvaluator  # noqa: F401

    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False


# Task definitions
TASK_TYPES = [
    "resistance",
    "escape",
    "binding",
    "immunogenicity",
    "stability",
    "codon_optimization",
]

# Disease-task mapping
DISEASE_TASKS = {
    "hiv": ["resistance", "escape", "binding", "immunogenicity"],
    "cancer": ["resistance", "binding", "immunogenicity", "stability"],
    "ra": ["binding", "immunogenicity", "stability"],
    "neuro": ["binding", "stability", "codon_optimization"],
    "flu": ["escape", "binding", "immunogenicity"],
    "bacterial": ["resistance", "binding"],
}


@dataclass
class MultiTaskConfig:
    """Configuration for multi-task training."""

    # Model architecture
    encoder_dim: int = 256
    hidden_dim: int = 128
    n_encoder_layers: int = 4
    n_decoder_layers: int = 2
    n_heads: int = 8
    dropout: float = 0.1

    # GradNorm parameters
    use_gradnorm: bool = True
    gradnorm_alpha: float = 1.5
    gradnorm_lr: float = 0.025

    # Training
    batch_size: int = 64
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5

    # Hardware
    use_amp: bool = True

    # Tasks and diseases
    diseases: list[str] = field(default_factory=lambda: ["hiv", "cancer", "ra", "neuro"])
    tasks: list[str] = field(default_factory=lambda: TASK_TYPES)


class SharedEncoder(nn.Module):
    """Shared transformer encoder for all tasks."""

    def __init__(
        self,
        input_dim: int = 64,  # Codon vocabulary
        hidden_dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 300,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Token embedding
        self.embed = nn.Embedding(input_dim, hidden_dim)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_dim)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Pool to sequence representation
        self.pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode sequence to representation."""
        batch_size, seq_len = x.shape

        # Embed
        token_emb = self.embed(x)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embed(positions)
        h = token_emb + pos_emb

        # Transform
        h = self.transformer(h, src_key_padding_mask=mask)

        # Pool (mean over non-masked positions)
        if mask is not None:
            mask_expanded = (~mask).unsqueeze(-1).float()
            h = (h * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            h = h.mean(dim=1)

        # Final projection
        h = self.pool(h)

        return h


class TaskHead(nn.Module):
    """Task-specific prediction head."""

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        output_dim: int = 1,
        n_layers: int = 2,
        task_type: str = "regression",
    ):
        super().__init__()
        self.task_type = task_type

        layers = []
        current_dim = input_dim
        for _ in range(n_layers - 1):
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.1),
                ]
            )
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class MultiTaskPredictor(nn.Module):
    """Multi-task prediction model with shared encoder and task-specific heads."""

    def __init__(self, config: MultiTaskConfig):
        super().__init__()
        self.config = config

        # Shared encoder
        self.encoder = SharedEncoder(
            input_dim=64,  # Codon vocabulary
            hidden_dim=config.encoder_dim,
            n_layers=config.n_encoder_layers,
            n_heads=config.n_heads,
            dropout=config.dropout,
        )

        # Disease-specific adapters
        self.disease_adapters = nn.ModuleDict()
        for disease in config.diseases:
            self.disease_adapters[disease] = nn.Sequential(
                nn.Linear(config.encoder_dim, config.encoder_dim),
                nn.GELU(),
                nn.Linear(config.encoder_dim, config.encoder_dim),
            )

        # Task heads (shared across diseases)
        self.task_heads = nn.ModuleDict()
        for task in config.tasks:
            output_dim = 1 if task != "codon_optimization" else 64
            self.task_heads[task] = TaskHead(
                input_dim=config.encoder_dim,
                hidden_dim=config.hidden_dim,
                output_dim=output_dim,
                n_layers=config.n_decoder_layers,
            )

        # GradNorm task weights
        if config.use_gradnorm:
            n_task_combos = sum(len(DISEASE_TASKS.get(d, [])) for d in config.diseases)
            self.task_weights = nn.Parameter(torch.ones(n_task_combos))
            self.initial_losses: dict | None = None

    def forward(
        self,
        x: torch.Tensor,
        disease: str,
        tasks: list[str] | None = None,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for specific disease and tasks."""
        # Encode
        h = self.encoder(x, mask)

        # Disease-specific adaptation
        if disease in self.disease_adapters:
            h = h + self.disease_adapters[disease](h)  # Residual

        # Task predictions
        if tasks is None:
            tasks = DISEASE_TASKS.get(disease, self.config.tasks)

        outputs = {}
        for task in tasks:
            if task in self.task_heads:
                outputs[task] = self.task_heads[task](h)

        return outputs

    def get_task_weight(self, disease: str, task: str) -> torch.Tensor:
        """Get weight for disease-task combination."""
        if not self.config.use_gradnorm:
            return torch.tensor(1.0)

        # Create deterministic index for disease-task pair
        idx = 0
        for d in sorted(self.config.diseases):
            for t in sorted(DISEASE_TASKS.get(d, [])):
                if d == disease and t == task:
                    return self.task_weights[idx]
                idx += 1

        return torch.tensor(1.0)


class GradNorm:
    """GradNorm for multi-task learning weight balancing."""

    def __init__(
        self,
        model: MultiTaskPredictor,
        alpha: float = 1.5,
        lr: float = 0.025,
    ):
        self.model = model
        self.alpha = alpha
        self.lr = lr
        self.initial_losses: dict = {}

    def update_weights(
        self,
        current_losses: dict[str, float],
        shared_params: list[nn.Parameter],
    ):
        """Update task weights based on GradNorm."""
        if not hasattr(self.model, "task_weights"):
            return

        # Store initial losses
        if not self.initial_losses:
            self.initial_losses = current_losses.copy()
            return

        # Compute loss ratios
        loss_ratios = {}
        for key, loss in current_losses.items():
            if key in self.initial_losses:
                loss_ratios[key] = loss / (self.initial_losses[key] + 1e-8)

        if not loss_ratios:
            return

        # Average loss ratio
        avg_ratio = sum(loss_ratios.values()) / len(loss_ratios)

        # Compute relative inverse training rates
        target_weights = {}
        for key, ratio in loss_ratios.items():
            target_weights[key] = (ratio / avg_ratio) ** self.alpha

        # Update weights (simplified gradient-free version)
        idx = 0
        for d in sorted(self.model.config.diseases):
            for t in sorted(DISEASE_TASKS.get(d, [])):
                key = f"{d}_{t}"
                if key in target_weights:
                    current = self.model.task_weights.data[idx].item()
                    target = target_weights[key]
                    # Exponential moving average update
                    self.model.task_weights.data[idx] = current + self.lr * (target - current)
                idx += 1

        # Normalize weights
        self.model.task_weights.data = (
            self.model.task_weights.data / self.model.task_weights.data.mean() * len(self.model.task_weights)
        )


class MultiDiseaseDataset(Dataset):
    """Dataset for multi-disease multi-task learning."""

    def __init__(
        self,
        diseases: list[str],
        max_seq_len: int = 300,
        split: str = "train",
    ):
        self.diseases = diseases
        self.max_seq_len = max_seq_len
        self.data = []
        self._load_data()

    def _load_data(self):
        """Load data from all diseases."""
        for disease in self.diseases:
            # Try to load real data
            data_paths = [
                PROJECT_ROOT / f"data/{disease}/labeled.pt",
                PROJECT_ROOT / f"research/bioinformatics/{disease}/data/labeled.pt",
            ]

            loaded = False
            for path in data_paths:
                if path.exists():
                    data = torch.load(path, weights_only=True)
                    for item in data:
                        item["disease"] = disease
                        self.data.append(item)
                    loaded = True
                    break

            if not loaded:
                # Generate synthetic data
                self._generate_synthetic(disease)

    def _generate_synthetic(self, disease: str):
        """Generate synthetic labeled data for testing."""
        n_samples = 500
        tasks = DISEASE_TASKS.get(disease, TASK_TYPES[:3])

        for _ in range(n_samples):
            length = torch.randint(50, self.max_seq_len, (1,)).item()
            seq = torch.randint(0, 64, (length,), dtype=torch.long)

            labels = {}
            for task in tasks:
                if task == "codon_optimization":
                    labels[task] = torch.randint(0, 64, (length,), dtype=torch.long)
                else:
                    labels[task] = torch.rand(1).item()

            self.data.append(
                {
                    "sequence": seq,
                    "disease": disease,
                    "labels": labels,
                }
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        seq = item["sequence"]

        # Pad/truncate
        if len(seq) > self.max_seq_len:
            seq = seq[: self.max_seq_len]
            mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        else:
            pad_len = self.max_seq_len - len(seq)
            mask = torch.cat([torch.zeros(len(seq), dtype=torch.bool), torch.ones(pad_len, dtype=torch.bool)])
            seq = torch.cat([seq, torch.zeros(pad_len, dtype=torch.long)])

        return {
            "sequence": seq,
            "mask": mask,
            "disease": item["disease"],
            "labels": item["labels"],
        }


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate for multi-task batch."""
    return {
        "sequence": torch.stack([item["sequence"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
        "diseases": [item["disease"] for item in batch],
        "labels": [item["labels"] for item in batch],
    }


def compute_task_loss(
    predictions: dict[str, torch.Tensor],
    labels: dict[str, float],
    task: str,
) -> torch.Tensor:
    """Compute loss for a single task."""
    if task not in predictions or task not in labels:
        return torch.tensor(0.0)

    pred = predictions[task]
    target = labels[task]

    if task == "codon_optimization":
        # Classification loss - prediction is single vector (batch, 64)
        # target is sequence of codons (seq_len,) - use first codon as target
        if isinstance(target, torch.Tensor):
            # Handle shape mismatch: pred is (batch, 64), target is (seq_len,)
            if target.numel() > 1:
                target = target[0:1]  # Use first codon as representative
            # Ensure target is on same device as prediction
            target = target.to(pred.device).long()
            return F.cross_entropy(pred.view(-1, 64), target.view(-1))
        return torch.tensor(0.0, device=pred.device)
    else:
        # Regression loss
        target_tensor = torch.tensor([target], device=pred.device, dtype=pred.dtype)
        return F.mse_loss(pred.squeeze(), target_tensor.squeeze())


def train_epoch(
    model: MultiTaskPredictor,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    gradnorm: GradNorm | None,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: MultiTaskConfig,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_losses = {}
    n_samples = 0

    for batch in dataloader:
        sequences = batch["sequence"].to(device)
        masks = batch["mask"].to(device)
        diseases = batch["diseases"]
        labels = batch["labels"]

        batch_loss = torch.tensor(0.0, device=device)

        with torch.amp.autocast("cuda", enabled=config.use_amp):
            # Process each sample (diseases may differ within batch)
            for i, (disease, sample_labels) in enumerate(zip(diseases, labels, strict=False)):
                seq = sequences[i : i + 1]
                mask = masks[i : i + 1]

                # Get predictions
                predictions = model(seq, disease, mask=mask)

                # Compute weighted losses
                for task, pred in predictions.items():
                    if task in sample_labels:
                        task_loss = compute_task_loss({task: pred}, sample_labels, task)
                        weight = model.get_task_weight(disease, task)
                        batch_loss = batch_loss + weight * task_loss

                        # Track losses
                        key = f"{disease}_{task}"
                        if key not in total_losses:
                            total_losses[key] = 0.0
                        total_losses[key] += task_loss.item()

        # Backward
        scaler.scale(batch_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        n_samples += len(diseases)

    # Normalize losses
    for key in total_losses:
        total_losses[key] /= n_samples

    # Update GradNorm weights
    if gradnorm is not None:
        shared_params = list(model.encoder.parameters())
        gradnorm.update_weights(total_losses, shared_params)

    return total_losses


@torch.no_grad()
def evaluate(
    model: MultiTaskPredictor,
    dataloader: DataLoader,
    device: torch.device,
    config: MultiTaskConfig,
) -> dict:
    """Evaluate model."""
    model.eval()
    total_losses = {}
    n_samples = 0

    for batch in dataloader:
        sequences = batch["sequence"].to(device)
        masks = batch["mask"].to(device)
        diseases = batch["diseases"]
        labels = batch["labels"]

        with torch.amp.autocast("cuda", enabled=config.use_amp):
            for i, (disease, sample_labels) in enumerate(zip(diseases, labels, strict=False)):
                seq = sequences[i : i + 1]
                mask = masks[i : i + 1]

                predictions = model(seq, disease, mask=mask)

                for task, pred in predictions.items():
                    if task in sample_labels:
                        task_loss = compute_task_loss({task: pred}, sample_labels, task)
                        key = f"val_{disease}_{task}"
                        if key not in total_losses:
                            total_losses[key] = 0.0
                        total_losses[key] += task_loss.item()

        n_samples += len(diseases)

    for key in total_losses:
        total_losses[key] /= n_samples

    return total_losses


def main():
    parser = argparse.ArgumentParser(description="Multi-Task Disease Predictor Training")
    parser.add_argument("--diseases", nargs="+", default=["hiv", "cancer", "ra", "neuro"], help="Diseases to train on")
    parser.add_argument("--tasks", nargs="+", default=TASK_TYPES, help="Tasks to train")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--weight-decay", type=float, default=0.01, help="Weight decay (default: 0.01, try 0.1 for less overfitting)"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate (default: 0.1)")
    parser.add_argument("--no-gradnorm", action="store_true", help="Disable GradNorm")
    parser.add_argument("--quick", action="store_true", help="Quick test mode")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")

    # Early stopping options
    parser.add_argument("--early-stopping", action="store_true", help="Enable early stopping")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--min-delta", type=float, default=0.001, help="Minimum improvement to reset patience")

    # New SOTA enhancement options
    parser.add_argument(
        "--use-escape-head",
        action="store_true",
        help="Use EVEscape-inspired VariantEscapeHead for escape prediction",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run ProteinGym-style evaluation after training",
    )
    parser.add_argument(
        "--n-drug-classes",
        type=int,
        default=6,
        help="Number of drug classes for escape prediction (default: 6)",
    )
    parser.add_argument(
        "--n-antibody-classes",
        type=int,
        default=10,
        help="Number of antibody classes for escape prediction (default: 10)",
    )
    args = parser.parse_args()

    # Check if enhanced features are available
    if (args.use_escape_head or args.evaluate) and not ENHANCED_AVAILABLE:
        print("\nWarning: Enhanced components not available. Using standard training.")
        args.use_escape_head = False
        args.evaluate = False

    print("\n" + "=" * 70)
    print("  MULTI-TASK DISEASE PREDICTOR TRAINING")
    print("=" * 70)
    print(f"  Diseases: {', '.join(args.diseases)}")
    print(f"  Tasks: {', '.join(args.tasks)}")
    print(f"  GradNorm: {'Disabled' if args.no_gradnorm else 'Enabled'}")
    print(f"  Weight Decay: {args.weight_decay}")
    print(f"  Dropout: {args.dropout}")
    if args.early_stopping:
        print(f"  Early Stopping: Enabled (patience={args.patience})")
    if args.use_escape_head:
        print("  Enhancement: EVEscape-inspired VariantEscapeHead")
    if args.evaluate:
        print("  Enhancement: ProteinGym-style evaluation")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")

    # Config
    config = MultiTaskConfig(
        epochs=10 if args.quick else args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        use_gradnorm=not args.no_gradnorm,
        diseases=args.diseases,
        tasks=args.tasks,
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    dataset = MultiDiseaseDataset(diseases=args.diseases)
    print(f"Dataset size: {len(dataset)}")

    # Split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn
    )

    # Model
    model = MultiTaskPredictor(config)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Create VariantEscapeHead if requested
    escape_head = None
    if args.use_escape_head and ENHANCED_AVAILABLE:
        print("\nCreating VariantEscapeHead (EVEscape-inspired)...")
        escape_head = VariantEscapeHead(
            latent_dim=config.encoder_dim,
            hidden_dim=config.hidden_dim,
            disease="hiv" if "hiv" in args.diseases else args.diseases[0],
            n_drug_classes=args.n_drug_classes,
            n_antibody_classes=args.n_antibody_classes,
            n_tcell_epitopes=20,
        )
        escape_head = escape_head.to(device)
        escape_params = sum(p.numel() for p in escape_head.parameters())
        print(f"  VariantEscapeHead parameters: {escape_params:,}")
        print(f"  Drug classes: {args.n_drug_classes}")
        print(f"  Antibody classes: {args.n_antibody_classes}")

    # Optimizer (include escape_head params if used)
    all_params = list(model.parameters())
    if escape_head is not None:
        all_params += list(escape_head.parameters())
    optimizer = AdamW(all_params, lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp)

    # GradNorm
    gradnorm = GradNorm(model, alpha=config.gradnorm_alpha, lr=config.gradnorm_lr) if config.use_gradnorm else None

    # Training
    checkpoint_dir = PROJECT_ROOT / "checkpoints/multitask_disease"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(config.epochs):
        train_losses = train_epoch(model, train_loader, optimizer, gradnorm, scaler, device, config)
        val_losses = evaluate(model, val_loader, device, config)
        scheduler.step()

        # Aggregate losses
        train_total = sum(train_losses.values())
        val_total = sum(val_losses.values())

        print(f"Epoch {epoch + 1}/{config.epochs} | Train Loss: {train_total:.4f} | Val Loss: {val_total:.4f}")

        # Print task weights if using GradNorm
        if config.use_gradnorm and (epoch + 1) % 10 == 0:
            weights = model.task_weights.data.cpu().numpy()
            print(f"  Task weights (sample): {weights[:5]}")

        # Save best and check early stopping
        if val_total < best_val_loss - args.min_delta:
            best_val_loss = val_total
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "val_loss": best_val_loss,
                },
                checkpoint_dir / "best.pt",
            )
            print("  [BEST] Saved checkpoint")
        else:
            patience_counter += 1

        # Early stopping check
        if args.early_stopping and patience_counter >= args.patience:
            print(f"\n  [EARLY STOP] No improvement for {args.patience} epochs. Best was epoch {best_epoch + 1}.")
            break

    # Save final
    save_dict = {
        "epoch": config.epochs,
        "model_state_dict": model.state_dict(),
        "config": config,
        "enhancements": {
            "escape_head": args.use_escape_head,
            "n_drug_classes": args.n_drug_classes,
            "n_antibody_classes": args.n_antibody_classes,
        },
    }
    if escape_head is not None:
        save_dict["escape_head_state_dict"] = escape_head.state_dict()
    torch.save(save_dict, checkpoint_dir / "latest.pt")

    # Run escape prediction on test samples if escape_head is used
    if escape_head is not None:
        print("\n" + "=" * 60)
        print("  VARIANT ESCAPE PREDICTION (EVEscape-Inspired)")
        print("=" * 60)
        model.eval()
        escape_head.eval()

        with torch.no_grad():
            # Get some validation samples
            val_samples = []
            for batch in val_loader:
                val_samples.append(batch["sequence"])
                if len(val_samples) >= 2:
                    break
            val_seqs = torch.cat(val_samples, dim=0)[:32].to(device)

            # Get encoder representations
            h = model.encoder(val_seqs)

            # Predict escape
            escape_preds = escape_head(h, return_components=True)

            print(f"\nEscape predictions on {val_seqs.shape[0]} samples:")
            for key, value in escape_preds.items():
                if value.dim() == 1 or value.shape[-1] == 1:
                    print(f"  {key}: mean={value.mean():.4f}, std={value.std():.4f}")
                else:
                    print(f"  {key}: shape={value.shape}")

    # Run ProteinGym-style evaluation if requested
    if args.evaluate and ENHANCED_AVAILABLE:
        print("\n" + "=" * 60)
        print("  PROTEINGYM-STYLE EVALUATION")
        print("=" * 60)

        # Get training sequences
        train_seqs = torch.stack([train_dataset[i]["sequence"] for i in range(min(500, len(train_dataset)))])

        # Get validation sequences for evaluation
        val_seqs = torch.stack([val_dataset[i]["sequence"] for i in range(min(200, len(val_dataset)))])

        evaluator = ProteinGymEvaluator(training_sequences=train_seqs)
        metrics = evaluator.evaluate(val_seqs)

        print(f"\nEvaluated {metrics.n_sequences} sequences:")
        print(f"  Quality - Mean tAI: {metrics.quality.mean_tai:.4f}")
        print(f"  Quality - Mean CAI: {metrics.quality.mean_cai:.4f}")
        print(f"  Novelty - Unique: {metrics.novelty.unique_fraction:.4f}")
        print(f"  Novelty - Novel: {metrics.novelty.novel_fraction:.4f}")
        print(f"  Diversity - Pairwise dist: {metrics.diversity.mean_pairwise_distance:.4f}")
        print(f"  Validity - No stops: {metrics.validity.no_stop_codons:.4f}")
        print("=" * 60)

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Checkpoint saved to: {checkpoint_dir}")
    if args.use_escape_head:
        print("  VariantEscapeHead: Enabled and saved")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
