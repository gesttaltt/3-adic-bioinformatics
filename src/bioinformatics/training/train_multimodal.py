# Copyright 2024-2026 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Training pipeline for multimodal DDG VAE.

This module provides training for the multimodal fusion VAE that
combines embeddings from all three specialist VAEs.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset

from src.bioinformatics.models.ddg_vae import DDGVAE
from src.bioinformatics.models.multimodal_ddg_vae import (
    MultimodalConfig,
    MultimodalDDGVAE,
)
from src.bioinformatics.training.deterministic import (
    DeterministicConfig,
    DeterministicTrainer,
)


@dataclass
class MultimodalTrainingConfig:
    """Configuration for multimodal VAE training."""

    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    # Multi-task loss weights
    consistency_weight: float = 0.1  # Cross-encoder agreement
    ranking_weight: float = 0.1  # Pairwise DDG ordering

    # Validation
    val_ratio: float = 0.2
    early_stopping_patience: int = 20

    # Checkpointing
    save_best: bool = True

    # Reproducibility
    deterministic: DeterministicConfig = None

    def __post_init__(self):
        if self.deterministic is None:
            self.deterministic = DeterministicConfig()


class MultimodalDataset(Dataset):
    """Dataset that provides inputs for all three specialist VAEs.

    Wraps three separate datasets and aligns them by mutation.
    """

    def __init__(
        self,
        s669_dataset: Dataset,
        protherm_dataset: Dataset,
        wide_dataset: Dataset,
    ):
        """Initialize multimodal dataset.

        Args:
            s669_dataset: S669 specialist dataset
            protherm_dataset: ProTherm specialist dataset
            wide_dataset: Wide specialist dataset
        """
        # For simplicity, use minimum length
        # In practice, you'd align by mutation
        self.length = min(
            len(s669_dataset),
            len(protherm_dataset),
            len(wide_dataset),
        )
        self.s669 = s669_dataset
        self.protherm = protherm_dataset
        self.wide = wide_dataset

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict:
        x_s669, y_s669 = self.s669[idx % len(self.s669)]
        x_protherm, y_protherm = self.protherm[idx % len(self.protherm)]
        x_wide, y_wide = self.wide[idx % len(self.wide)]

        # Use S669 labels as ground truth (benchmark)
        return {
            "x_s669": x_s669,
            "x_protherm": x_protherm,
            "x_wide": x_wide,
            "y": y_s669,
        }


class MultimodalTrainer(DeterministicTrainer):
    """Trainer for multimodal DDG VAE.

    Trains the fusion layer and decoder while keeping specialist
    encoders frozen.
    """

    def __init__(
        self,
        vae_s669: DDGVAE,
        vae_protherm: DDGVAE,
        vae_wide: DDGVAE,
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
        config: MultimodalTrainingConfig | None = None,
        multimodal_config: MultimodalConfig | None = None,
        device: str = "cuda",
        output_dir: Path | None = None,
    ):
        """Initialize multimodal trainer.

        Args:
            vae_s669: Pre-trained S669 specialist
            vae_protherm: Pre-trained ProTherm specialist
            vae_wide: Pre-trained Wide specialist
            train_dataset: Training dataset (MultimodalDataset)
            val_dataset: Validation dataset
            config: MultimodalTrainingConfig
            multimodal_config: MultimodalConfig for fusion
            device: Training device
            output_dir: Output directory
        """
        if config is None:
            config = MultimodalTrainingConfig()

        super().__init__(config.deterministic)

        self.config = config
        self.device = device

        # Create multimodal VAE
        self.model = MultimodalDDGVAE(
            vae_s669=vae_s669,
            vae_protherm=vae_protherm,
            vae_wide=vae_wide,
            config=multimodal_config,
            freeze_specialists=True,
        ).to(device)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        if output_dir is None:
            output_dir = Path("outputs/multimodal_ddg")
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

        # Optimizer (only fusion and decoder parameters)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=1e-6,
        )

    def _ranking_loss(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute pairwise ranking loss.

        Encourages correct relative ordering of DDG predictions.
        """
        # All pairs
        n = preds.shape[0]
        if n < 2:
            return torch.tensor(0.0, device=preds.device)

        pred_diff = preds.view(-1, 1) - preds.view(1, -1)
        target_diff = targets.view(-1, 1) - targets.view(1, -1)

        # Margin ranking loss
        margin = 0.1
        loss = torch.relu(margin - pred_diff * torch.sign(target_diff))
        return loss.mean()

    def train_epoch(self) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            x_s669 = batch["x_s669"].to(self.device)
            x_protherm = batch["x_protherm"].to(self.device)
            x_wide = batch["x_wide"].to(self.device)
            y = batch["y"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            loss_dict = self.model.loss(x_s669, x_protherm, x_wide, y)
            loss = loss_dict["loss"]

            # Add ranking loss
            if self.config.ranking_weight > 0:
                output = self.model(x_s669, x_protherm, x_wide)
                preds = output["ddg_pred"].squeeze()
                ranking_loss = self._ranking_loss(preds, y)
                loss = loss + self.config.ranking_weight * ranking_loss

            loss.backward()

            if self.config.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            self.optimizer.step()

            total_loss += loss.item()
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
            x_s669 = batch["x_s669"].to(self.device)
            x_protherm = batch["x_protherm"].to(self.device)
            x_wide = batch["x_wide"].to(self.device)
            y = batch["y"].to(self.device)

            preds = self.model.predict(x_s669, x_protherm, x_wide)
            all_preds.extend(preds.squeeze().cpu().numpy())
            all_targets.extend(y.cpu().numpy())

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

            if epoch % 10 == 0:
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

        path = self.output_dir / "final.pt"
        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "best_metric": self.best_metric,
            },
            path,
        )


__all__ = [
    "MultimodalTrainingConfig",
    "MultimodalDataset",
    "MultimodalTrainer",
]
