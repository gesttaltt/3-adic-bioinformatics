# Copyright 2024-2026 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Training pipeline for single-dataset DDG VAE.

This module provides the trainer for individual specialist VAEs:
- VAE-S669: Trained on S669 benchmark
- VAE-ProTherm: Trained on curated ProTherm data
- VAE-Wide: Trained on ProteinGym + other sources
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import Dataset

from src.bioinformatics.models.ddg_vae import DDGVAE
from src.bioinformatics.training.deterministic import (
    DeterministicConfig,
    DeterministicTrainer,
)


@dataclass
class TrainingConfig:
    """Configuration for DDG VAE training."""

    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    # Scheduler
    scheduler: str = "cosine"  # "cosine", "plateau", or "none"
    scheduler_patience: int = 10
    min_lr: float = 1e-6

    # Validation
    val_ratio: float = 0.2
    val_every: int = 1
    early_stopping_patience: int = 20
    early_stopping_metric: str = "val_spearman"

    # Checkpointing
    save_every: int = 10
    keep_last: int = 5
    save_best: bool = True

    # Logging
    log_every: int = 10
    verbose: bool = True

    # VAE-specific
    beta_schedule: str = "constant"  # "constant", "warmup", "cyclic"
    beta_warmup_epochs: int = 10

    # Reproducibility
    deterministic: DeterministicConfig = field(default_factory=DeterministicConfig)


class DDGVAETrainer(DeterministicTrainer):
    """Trainer for DDG VAE models.

    Supports training VAE-S669, VAE-ProTherm, and VAE-Wide variants
    with appropriate configurations.
    """

    def __init__(
        self,
        model: DDGVAE,
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
        config: TrainingConfig | None = None,
        device: str = "cuda",
        output_dir: Path | None = None,
    ):
        """Initialize trainer.

        Args:
            model: DDGVAE model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            config: TrainingConfig
            device: Training device
            output_dir: Directory for outputs
        """
        if config is None:
            config = TrainingConfig()

        super().__init__(config.deterministic)

        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        # Store training config separately to avoid overwriting parent's self.config
        self.training_config = config
        self.device = device

        if output_dir is None:
            output_dir = Path("outputs/ddg_vae")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.epoch = 0
        self.best_metric = float("-inf")
        self.epochs_without_improvement = 0
        self.history: dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_spearman": [],
            "val_pearson": [],
        }

        # Set up determinism
        self.setup_determinism()

        # Create data loaders
        self.train_loader = self.create_dataloader(train_dataset, self.training_config.batch_size, shuffle=True)
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = self.create_dataloader(val_dataset, self.training_config.batch_size, shuffle=False)

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )

        # Scheduler
        self.scheduler = self._create_scheduler()

    def _create_scheduler(self) -> object | None:
        """Create learning rate scheduler."""
        if self.training_config.scheduler == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.training_config.epochs,
                eta_min=self.training_config.min_lr,
            )
        elif self.training_config.scheduler == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=0.5,
                patience=self.training_config.scheduler_patience,
                min_lr=self.training_config.min_lr,
            )
        return None

    def _get_beta(self, epoch: int) -> float:
        """Get KL weight for current epoch."""
        base_beta = self.model.config.beta

        if self.training_config.beta_schedule == "constant":
            return base_beta
        elif self.training_config.beta_schedule == "warmup":
            warmup = self.training_config.beta_warmup_epochs
            if epoch < warmup:
                return base_beta * (epoch + 1) / warmup
            return base_beta
        elif self.training_config.beta_schedule == "cyclic":
            cycle_length = 20
            phase = epoch % cycle_length
            return base_beta * (0.5 + 0.5 * np.cos(np.pi * phase / cycle_length))
        return base_beta

    def train_epoch(self) -> dict[str, float]:
        """Train for one epoch.

        Returns:
            Dictionary with epoch metrics
        """
        self.model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        n_batches = 0

        # Adjust beta for this epoch
        original_beta = self.model.config.beta
        self.model.config.beta = self._get_beta(self.epoch)

        for _batch_idx, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            loss_dict = self.model.loss(x, y)
            loss = loss_dict["loss"]

            loss.backward()

            if self.training_config.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.grad_clip)

            self.optimizer.step()

            total_loss += loss.item()
            total_recon += loss_dict["recon_loss"].item()
            total_kl += loss_dict["kl_loss"].item()
            n_batches += 1

        # Restore original beta
        self.model.config.beta = original_beta

        return {
            "train_loss": total_loss / n_batches,
            "train_recon": total_recon / n_batches,
            "train_kl": total_kl / n_batches,
            "beta": self._get_beta(self.epoch),
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Run validation.

        Returns:
            Dictionary with validation metrics
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        n_batches = 0

        for x, y in self.val_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            loss_dict = self.model.loss(x, y)
            preds = self.model.predict(x)

            total_loss += loss_dict["loss"].item()
            # Ensure both preds and targets are flattened to 1D
            all_preds.extend(preds.squeeze(-1).cpu().numpy().flatten())
            all_targets.extend(y.cpu().numpy().flatten())
            n_batches += 1

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        spearman_r, spearman_p = spearmanr(all_preds, all_targets)
        pearson_r, pearson_p = pearsonr(all_preds, all_targets)
        mae = np.mean(np.abs(all_preds - all_targets))
        rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))

        return {
            "val_loss": total_loss / n_batches,
            "val_spearman": spearman_r,
            "val_spearman_p": spearman_p,
            "val_pearson": pearson_r,
            "val_pearson_p": pearson_p,
            "val_mae": mae,
            "val_rmse": rmse,
        }

    def train(
        self,
        callback: Callable[[int, dict], None] | None = None,
    ) -> dict:
        """Run full training loop.

        Args:
            callback: Optional callback function(epoch, metrics)

        Returns:
            Final metrics dictionary
        """
        for epoch in range(self.training_config.epochs):
            self.epoch = epoch

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = {}
            if epoch % self.training_config.val_every == 0 and self.val_loader is not None:
                val_metrics = self.validate()

            # Update history
            self.history["train_loss"].append(train_metrics["train_loss"])
            if val_metrics:
                self.history["val_loss"].append(val_metrics.get("val_loss", 0))
                self.history["val_spearman"].append(val_metrics.get("val_spearman", 0))
                self.history["val_pearson"].append(val_metrics.get("val_pearson", 0))

            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau) and val_metrics:
                    self.scheduler.step(val_metrics.get("val_spearman", 0))
                elif isinstance(self.scheduler, CosineAnnealingLR):
                    self.scheduler.step()

            # Log
            if self.training_config.verbose and epoch % self.training_config.log_every == 0:
                self._log_epoch(epoch, train_metrics, val_metrics)

            # Callback
            if callback is not None:
                callback(epoch, {**train_metrics, **val_metrics})

            # Checkpointing
            if (epoch + 1) % self.training_config.save_every == 0:
                self._save_checkpoint(epoch, train_metrics, val_metrics)

            # Early stopping
            current_metric = val_metrics.get(self.training_config.early_stopping_metric, 0)
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.epochs_without_improvement = 0
                if self.training_config.save_best:
                    self._save_best(epoch, train_metrics, val_metrics)
            else:
                self.epochs_without_improvement += 1

            if self.epochs_without_improvement >= self.training_config.early_stopping_patience:
                if self.training_config.verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

        # Save final state
        self._save_final()

        return {
            "best_metric": self.best_metric,
            "final_epoch": self.epoch,
            "history": self.history,
        }

    def _log_epoch(
        self,
        epoch: int,
        train_metrics: dict,
        val_metrics: dict,
    ) -> None:
        """Log epoch metrics."""
        msg = f"Epoch {epoch:3d}: "
        msg += f"loss={train_metrics['train_loss']:.4f} "
        if val_metrics:
            msg += f"val_loss={val_metrics.get('val_loss', 0):.4f} "
            msg += f"spearman={val_metrics.get('val_spearman', 0):.4f} "
        print(msg)

    def _save_checkpoint(
        self,
        epoch: int,
        train_metrics: dict,
        val_metrics: dict,
    ) -> None:
        """Save training checkpoint."""
        path = self.output_dir / f"checkpoint_epoch{epoch:03d}.pt"
        self.save_checkpoint(
            str(path),
            self.model,
            self.optimizer,
            epoch,
            {**train_metrics, **val_metrics},
        )

    def _save_best(
        self,
        epoch: int,
        train_metrics: dict,
        val_metrics: dict,
    ) -> None:
        """Save best model."""
        path = self.output_dir / "best.pt"
        self.save_checkpoint(
            str(path),
            self.model,
            self.optimizer,
            epoch,
            {**train_metrics, **val_metrics},
        )

    def _save_final(self) -> None:
        """Save final training state."""
        # Save history (convert numpy floats to Python floats for JSON)
        history_path = self.output_dir / "training_history.json"
        serializable_history = {k: [float(v) for v in vals] for k, vals in self.history.items()}
        with open(history_path, "w") as f:
            json.dump(serializable_history, f, indent=2)

        # Save final checkpoint
        path = self.output_dir / "final.pt"
        self.save_checkpoint(
            str(path),
            self.model,
            self.optimizer,
            self.epoch,
            {"best_metric": self.best_metric},
        )


def train_vae_s669(
    dataset: Dataset,
    output_dir: Path,
    config: TrainingConfig | None = None,
    use_hyperbolic: bool = True,
    device: str = "cuda",
) -> DDGVAE:
    """Train VAE-S669 specialist.

    Args:
        dataset: S669 dataset
        output_dir: Output directory
        config: Training configuration
        use_hyperbolic: Use hyperbolic features
        device: Training device

    Returns:
        Trained DDGVAE model
    """
    model = DDGVAE.create_s669_variant(use_hyperbolic=use_hyperbolic)

    # Default config for S669
    if config is None:
        config = TrainingConfig(
            epochs=100,
            batch_size=32,
            learning_rate=1e-4,
            early_stopping_patience=20,
        )

    # Split dataset
    from torch.utils.data import random_split

    n_val = int(len(dataset) * config.val_ratio)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(config.deterministic.seed)
    )

    trainer = DDGVAETrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=config,
        device=device,
        output_dir=output_dir,
    )

    trainer.train()
    return model


def train_vae_protherm(
    dataset: Dataset,
    output_dir: Path,
    config: TrainingConfig | None = None,
    use_hyperbolic: bool = True,
    device: str = "cuda",
) -> DDGVAE:
    """Train VAE-ProTherm specialist."""
    model = DDGVAE.create_protherm_variant(use_hyperbolic=use_hyperbolic)

    if config is None:
        config = TrainingConfig(
            epochs=200,  # More epochs for curated data
            batch_size=16,  # Smaller batch for smaller dataset
            learning_rate=5e-5,  # Slower for cleaner data
            early_stopping_patience=30,
        )

    from torch.utils.data import random_split

    n_val = int(len(dataset) * config.val_ratio)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(config.deterministic.seed)
    )

    trainer = DDGVAETrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=config,
        device=device,
        output_dir=output_dir,
    )

    trainer.train()
    return model


def train_vae_wide(
    dataset: Dataset,
    output_dir: Path,
    config: TrainingConfig | None = None,
    use_hyperbolic: bool = True,
    device: str = "cuda",
) -> DDGVAE:
    """Train VAE-Wide specialist."""
    model = DDGVAE.create_wide_variant(use_hyperbolic=use_hyperbolic)

    if config is None:
        config = TrainingConfig(
            epochs=50,  # Fewer epochs for large data
            batch_size=128,  # Larger batch for large dataset
            learning_rate=1e-3,  # Faster for diverse data
            early_stopping_patience=10,
        )

    from torch.utils.data import random_split

    n_val = int(len(dataset) * config.val_ratio)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(config.deterministic.seed)
    )

    trainer = DDGVAETrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=config,
        device=device,
        output_dir=output_dir,
    )

    trainer.train()
    return model


__all__ = [
    "TrainingConfig",
    "DDGVAETrainer",
    "train_vae_s669",
    "train_vae_protherm",
    "train_vae_wide",
]
