# Copyright 2024-2026 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Training pipeline for DDG transformer models.

This module provides training for:
- DDGTransformer: Full-sequence transformer
- HierarchicalTransformer: Two-level attention
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset

from src.bioinformatics.models.ddg_transformer import (
    DDGTransformer,
    HierarchicalTransformer,
    TransformerConfig,
)
from src.bioinformatics.training.deterministic import (
    DeterministicConfig,
    DeterministicTrainer,
)

# Amino acid vocabulary
AA_VOCAB = {
    "A": 0,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "V": 17,
    "W": 18,
    "Y": 19,
    "-": 20,  # Gap
    "*": 21,  # Mask/unknown
}


def tokenize_sequence(sequence: str) -> list[int]:
    """Convert amino acid sequence to token indices."""
    return [AA_VOCAB.get(aa.upper(), AA_VOCAB["*"]) for aa in sequence]


@dataclass
class TransformerTrainingConfig:
    """Configuration for transformer training."""

    # Training parameters
    epochs: int = 50
    batch_size: int = 4  # Small for memory
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    # Learning rate schedule
    warmup_epochs: int = 5
    min_lr: float = 1e-6

    # Memory optimization
    accumulation_steps: int = 8  # Effective batch = batch_size * accumulation_steps
    use_mixed_precision: bool = True

    # Validation
    val_ratio: float = 0.2
    early_stopping_patience: int = 15

    # Checkpointing
    save_best: bool = True

    # Reproducibility
    deterministic: DeterministicConfig = field(default_factory=DeterministicConfig)


class SequenceDataset(Dataset):
    """Dataset with protein sequences for transformer training."""

    def __init__(
        self,
        sequences: list[str],
        mutation_positions: list[int],
        ddg_values: list[float],
        max_len: int = 256,
    ):
        """Initialize sequence dataset.

        Args:
            sequences: List of protein sequences
            mutation_positions: List of mutation positions (0-indexed)
            ddg_values: List of DDG values
            max_len: Maximum sequence length (will truncate/pad)
        """
        self.sequences = sequences
        self.mutation_positions = mutation_positions
        self.ddg_values = ddg_values
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        seq = self.sequences[idx]
        pos = self.mutation_positions[idx]
        ddg = self.ddg_values[idx]

        # Tokenize
        tokens = tokenize_sequence(seq)

        # Truncate or pad
        if len(tokens) > self.max_len:
            # Center around mutation position
            start = max(0, pos - self.max_len // 2)
            tokens = tokens[start : start + self.max_len]
            pos = pos - start
        else:
            # Pad with gaps
            tokens = tokens + [AA_VOCAB["-"]] * (self.max_len - len(tokens))

        # Create padding mask
        padding_mask = torch.zeros(self.max_len, dtype=torch.bool)
        padding_mask[len(seq) :] = True

        return {
            "sequence": torch.tensor(tokens, dtype=torch.long),
            "mutation_pos": torch.tensor(pos, dtype=torch.long),
            "padding_mask": padding_mask,
            "ddg": torch.tensor(ddg, dtype=torch.float32),
        }


class TransformerTrainer(DeterministicTrainer):
    """Trainer for DDG transformer models.

    Handles memory-constrained training with gradient accumulation
    and optional mixed precision.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
        config: TransformerTrainingConfig | None = None,
        device: str = "cuda",
        output_dir: Path | None = None,
    ):
        """Initialize transformer trainer.

        Args:
            model: DDGTransformer or HierarchicalTransformer
            train_dataset: Training dataset (SequenceDataset)
            val_dataset: Validation dataset
            config: TransformerTrainingConfig
            device: Training device
            output_dir: Output directory
        """
        if config is None:
            config = TransformerTrainingConfig()

        super().__init__(config.deterministic)

        self.model = model.to(device)
        self.config = config
        self.device = device
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        if output_dir is None:
            output_dir = Path("outputs/ddg_transformer")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.epoch = 0
        self.best_metric = float("-inf")
        self.epochs_without_improvement = 0
        self.history = {"train_loss": [], "val_spearman": []}

        # Set up determinism
        self.setup_determinism()

        # Create data loaders
        self.train_loader = self.create_dataloader(train_dataset, config.batch_size, shuffle=True)
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = self.create_dataloader(val_dataset, config.batch_size, shuffle=False)

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler with warm restarts
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.epochs // 4,
            T_mult=2,
            eta_min=config.min_lr,
        )

        # Mixed precision
        self.scaler = None
        if config.use_mixed_precision and torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler()

    def train_epoch(self) -> dict[str, float]:
        """Train for one epoch with gradient accumulation."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            sequence = batch["sequence"].to(self.device)
            mutation_pos = batch["mutation_pos"].to(self.device)
            padding_mask = batch["padding_mask"].to(self.device)
            ddg = batch["ddg"].to(self.device)

            # Forward pass with optional mixed precision
            if self.scaler is not None:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    output = self.model(sequence, mutation_pos, padding_mask)
                    loss = F.mse_loss(output["ddg_pred"].squeeze(), ddg)
                    loss = loss / self.config.accumulation_steps

                self.scaler.scale(loss).backward()
            else:
                output = self.model(sequence, mutation_pos, padding_mask)
                loss = F.mse_loss(output["ddg_pred"].squeeze(), ddg)
                loss = loss / self.config.accumulation_steps
                loss.backward()

            # Gradient accumulation step
            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                if self.config.grad_clip > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

            total_loss += loss.item() * self.config.accumulation_steps
            n_batches += 1

        return {
            "train_loss": total_loss / n_batches,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Run validation."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        all_preds = []
        all_targets = []

        for batch in self.val_loader:
            sequence = batch["sequence"].to(self.device)
            mutation_pos = batch["mutation_pos"].to(self.device)
            padding_mask = batch["padding_mask"].to(self.device)
            ddg = batch["ddg"].to(self.device)

            preds = self.model.predict(sequence, mutation_pos, padding_mask)
            all_preds.extend(preds.squeeze().cpu().numpy())
            all_targets.extend(ddg.cpu().numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        spearman_r, _ = spearmanr(all_preds, all_targets)
        pearson_r, _ = pearsonr(all_preds, all_targets)
        mae = np.mean(np.abs(all_preds - all_targets))

        return {
            "val_spearman": spearman_r,
            "val_pearson": pearson_r,
            "val_mae": mae,
        }

    def train(
        self,
        callback: Callable[[int, dict], None] | None = None,
    ) -> dict:
        """Run full training loop."""
        for epoch in range(self.config.epochs):
            self.epoch = epoch

            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            self.history["train_loss"].append(train_metrics["train_loss"])
            if val_metrics:
                self.history["val_spearman"].append(val_metrics.get("val_spearman", 0))

            self.scheduler.step()

            if epoch % 5 == 0:
                print(
                    f"Epoch {epoch:3d}: "
                    f"loss={train_metrics['train_loss']:.4f} "
                    f"spearman={val_metrics.get('val_spearman', 0):.4f}"
                )

            if callback is not None:
                callback(epoch, {**train_metrics, **val_metrics})

            # Early stopping
            current_metric = val_metrics.get("val_spearman", 0)
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.epochs_without_improvement = 0
                if self.config.save_best:
                    self._save_best(epoch, train_metrics, val_metrics)
            else:
                self.epochs_without_improvement += 1

            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

        self._save_final()

        return {
            "best_metric": self.best_metric,
            "final_epoch": self.epoch,
            "history": self.history,
        }

    def _save_best(self, epoch: int, train_metrics: dict, val_metrics: dict) -> None:
        """Save best model."""
        path = self.output_dir / "best.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": {**train_metrics, **val_metrics},
                "best_metric": self.best_metric,
            },
            path,
        )

    def _save_final(self) -> None:
        """Save final state."""
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)


def train_full_sequence_transformer(
    train_dataset: SequenceDataset,
    val_dataset: SequenceDataset | None = None,
    config: TransformerTrainingConfig | None = None,
    transformer_config: TransformerConfig | None = None,
    device: str = "cuda",
    output_dir: Path | None = None,
) -> DDGTransformer:
    """Train full-sequence DDG transformer.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Training configuration
        transformer_config: Transformer architecture config
        device: Training device
        output_dir: Output directory

    Returns:
        Trained DDGTransformer
    """
    if transformer_config is None:
        # Memory-optimized config for 6GB VRAM
        transformer_config = TransformerConfig(
            max_seq_len=256,
            d_model=128,
            n_heads=4,
            n_layers=3,
            use_gradient_checkpointing=True,
        )

    model = DDGTransformer(transformer_config)

    trainer = TransformerTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        device=device,
        output_dir=output_dir,
    )

    trainer.train()
    return model


def train_hierarchical_transformer(
    train_dataset: SequenceDataset,
    val_dataset: SequenceDataset | None = None,
    config: TransformerTrainingConfig | None = None,
    transformer_config: TransformerConfig | None = None,
    device: str = "cuda",
    output_dir: Path | None = None,
) -> HierarchicalTransformer:
    """Train hierarchical DDG transformer.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Training configuration
        transformer_config: Transformer architecture config
        device: Training device
        output_dir: Output directory

    Returns:
        Trained HierarchicalTransformer
    """
    if transformer_config is None:
        transformer_config = TransformerConfig(
            max_seq_len=256,
            d_model=128,
            n_heads=4,
            n_layers=3,
            local_window=21,
            stride=4,
            use_gradient_checkpointing=True,
        )

    model = HierarchicalTransformer(transformer_config)

    trainer = TransformerTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        device=device,
        output_dir=output_dir,
    )

    trainer.train()
    return model


__all__ = [
    "AA_VOCAB",
    "tokenize_sequence",
    "TransformerTrainingConfig",
    "SequenceDataset",
    "TransformerTrainer",
    "train_full_sequence_transformer",
    "train_hierarchical_transformer",
]
