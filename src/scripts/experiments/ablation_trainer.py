#!/usr/bin/env python
# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Ablation Trainer - Runs actual training for feature ablation studies.

This module provides the actual training loop for isolated feature testing.
Each experiment runs with a specific configuration, measuring the impact
of individual features.

Integration with parallel_feature_ablation.py:
    from scripts.experiments.ablation_trainer import run_ablation_training
    result = run_ablation_training(config)
"""

from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class AblationConfig:
    """Configuration for ablation experiment."""

    name: str
    seed: int = 42
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    latent_dim: int = 16

    # Feature flags
    enable_hyperbolic_prior: bool = False
    enable_curriculum_learning: bool = False
    enable_beta_warmup: bool = False
    enable_radial_stratification: bool = False
    enable_padic_ranking_loss: bool = False
    enable_fisher_rao_loss: bool = False

    # Regularization settings
    dropout: float = 0.0
    enable_early_stopping: bool = False
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.001

    # Model architecture
    hidden_dims: list | None = None  # None = default [64, 32]

    # Curriculum settings
    curriculum_tau_scale: float = 0.1
    curriculum_start_epoch: int = 0

    # Beta warmup settings
    beta_warmup_start_epoch: int = 50
    beta_warmup_initial: float = 0.0
    beta_warmup_epochs: int = 10

    # Radial stratification settings
    radial_inner: float = 0.1
    radial_outer: float = 0.85
    radial_weight: float = 0.3

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class AblationResult:
    """Results from ablation experiment."""

    name: str
    best_coverage: float = 0.0
    best_correlation: float = 0.0
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    final_coverage: float = 0.0
    final_correlation: float = 0.0
    final_loss: float = float("inf")
    training_time: float = 0.0
    had_loss_spike: bool = False
    max_spike_ratio: float = 1.0
    early_stopped: bool = False
    early_stop_epoch: int = 0
    total_epochs_run: int = 0
    completed: bool = False
    error: str | None = None


def set_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_ablation_training(config: AblationConfig) -> AblationResult:
    """Run a single ablation training experiment.

    This function runs the actual training loop with the specified
    configuration, measuring metrics for comparison.

    Args:
        config: Ablation experiment configuration

    Returns:
        AblationResult with all metrics
    """
    from src.training import EpochMetrics, GrokDetector

    set_seeds(config.seed)
    result = AblationResult(name=config.name)
    start_time = time.time()

    try:
        # Import training components
        from src.data.generation import generate_all_ternary_operations
        from src.training import TernaryDataset

        # ================================================================
        # DATA LOADING
        # ================================================================
        print(f"\n[{config.name}] Loading data...")

        # Generate ternary operations dataset (19683 operations)
        all_operations = generate_all_ternary_operations()
        operations_tensor = torch.tensor(all_operations, dtype=torch.float32)
        indices = torch.arange(len(all_operations))
        dataset = TernaryDataset(operations_tensor, indices)

        # Split into train/val
        n_total = len(dataset)
        n_val = int(0.1 * n_total)
        n_train = n_total - n_val

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [n_train, n_val], generator=torch.Generator().manual_seed(config.seed)
        )

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

        # ================================================================
        # MODEL CREATION
        # ================================================================
        print(f"[{config.name}] Creating model...")

        # Use SimpleVAE for ablation (fully trainable, no frozen components)
        from src.models.simple_vae import SimpleVAE, SimpleVAEWithHyperbolic

        hidden_dims = config.hidden_dims if config.hidden_dims else [64, 32]

        # Choose model based on hyperbolic prior setting
        if config.enable_hyperbolic_prior:
            model = SimpleVAEWithHyperbolic(
                input_dim=9,
                latent_dim=config.latent_dim,
                hidden_dims=hidden_dims,
                dropout=config.dropout,
            ).to(config.device)
            print(
                f"[{config.name}] Model: SimpleVAEWithHyperbolic, hidden={hidden_dims}, latent={config.latent_dim}, dropout={config.dropout}"
            )
        else:
            model = SimpleVAE(
                input_dim=9,
                latent_dim=config.latent_dim,
                hidden_dims=hidden_dims,
                dropout=config.dropout,
            ).to(config.device)
            print(
                f"[{config.name}] Model: SimpleVAE, hidden={hidden_dims}, latent={config.latent_dim}, dropout={config.dropout}"
            )

        # Print parameter count
        params = model.count_parameters()
        print(f"[{config.name}] Parameters: {params['trainable']:,} trainable")

        # ================================================================
        # OPTIMIZER
        # ================================================================
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # ================================================================
        # LOSS COMPONENTS
        # ================================================================
        # Use simple reconstruction + KL loss for ablation (not the full dual VAE loss)
        from src.losses.dual_vae_loss import KLDivergenceLoss, ReconstructionLoss

        recon_loss_fn = ReconstructionLoss()
        kl_loss_fn = KLDivergenceLoss()

        def loss_fn(logits, x, mu, logvar, beta=0.01):
            """Simple VAE loss: reconstruction (from logits) + beta * KL.

            Note: beta=1.0 is too strong for this dataset - kills learning.
            beta=0.01 allows reconstruction while still regularizing.
            """
            recon = recon_loss_fn(logits, x)
            kl = kl_loss_fn(mu, logvar)
            return recon + beta * kl

        # Optional: P-adic ranking loss (needs batch indices for 3-adic distance computation)
        padic_loss_fn = None
        if config.enable_padic_ranking_loss:
            from src.losses.padic import PAdicRankingLoss

            padic_loss_fn = PAdicRankingLoss()

        # Optional: Fisher-Rao regularization (use NaturalGradientRegularizer that takes mu, logvar)
        fisher_rao_fn = None
        if config.enable_fisher_rao_loss:
            from src.losses.fisher_rao import NaturalGradientRegularizer

            fisher_rao_fn = NaturalGradientRegularizer()

        # ================================================================
        # TRAINING LOOP
        # ================================================================
        print(f"[{config.name}] Starting training for {config.epochs} epochs...")
        if config.enable_early_stopping:
            print(f"[{config.name}] Early stopping enabled: patience={config.early_stopping_patience}")

        detector = GrokDetector()
        loss_history = []

        best_coverage = 0.0
        best_correlation = 0.0
        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        early_stopped = False
        epochs_run = 0

        for epoch in range(config.epochs):
            epochs_run = epoch + 1
            model.train()
            epoch_loss = 0.0

            # Beta schedule (default 0.01, beta=1.0 kills learning on this dataset)
            beta = 0.01
            if config.enable_beta_warmup:
                if epoch < config.beta_warmup_start_epoch:
                    beta = 1.0
                elif epoch < config.beta_warmup_start_epoch + config.beta_warmup_epochs:
                    progress = (epoch - config.beta_warmup_start_epoch) / config.beta_warmup_epochs
                    beta = config.beta_warmup_initial + (1.0 - config.beta_warmup_initial) * progress
                else:
                    beta = 1.0

            # Curriculum tau (used to scale difficulty/temperature)
            curriculum_tau = 1.0
            if config.enable_curriculum_learning and epoch >= config.curriculum_start_epoch:
                curriculum_tau = min(1.0, config.curriculum_tau_scale * (epoch - config.curriculum_start_epoch))

            for _batch_idx, batch in enumerate(train_loader):
                # Handle dict batch from TernaryDataset
                if isinstance(batch, dict):
                    x = batch["operation"].to(config.device)
                    batch_indices = batch["index"].to(config.device)
                elif isinstance(batch, (list, tuple)):
                    x = batch[0].to(config.device)
                    batch_indices = torch.arange(x.size(0), device=config.device)
                else:
                    x = batch.to(config.device)
                    batch_indices = torch.arange(x.size(0), device=config.device)

                optimizer.zero_grad()

                # Forward pass
                outputs = model(x)

                # SimpleVAE output format: dict with logits, mu, logvar, z
                recon_logits = outputs["logits"]
                mu = outputs["mu"]
                logvar = outputs["logvar"]
                z = outputs["z"]

                # Compute losses (curriculum_tau scales reconstruction importance)
                recon_weight = curriculum_tau if config.enable_curriculum_learning else 1.0
                loss = loss_fn(recon_logits, x, mu, logvar, beta=beta) * recon_weight

                # Optional losses
                if padic_loss_fn is not None:
                    padic_loss = padic_loss_fn(z, batch_indices)
                    loss = loss + 0.1 * padic_loss

                if fisher_rao_fn is not None:
                    fr_loss, _ = fisher_rao_fn(mu, logvar)
                    loss = loss + 0.05 * fr_loss

                # Radial stratification
                if config.enable_radial_stratification:
                    radius = torch.norm(z, dim=-1)
                    radial_penalty = torch.relu(radius - config.radial_outer) + torch.relu(config.radial_inner - radius)
                    loss = loss + config.radial_weight * radial_penalty.mean()

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= len(train_loader)
            loss_history.append(epoch_loss)

            # ================================================================
            # VALIDATION
            # ================================================================
            model.eval()
            val_loss = 0.0
            all_z = []

            with torch.no_grad():
                for batch in val_loader:
                    # Handle dict batch from TernaryDataset
                    if isinstance(batch, dict):
                        x = batch["operation"].to(config.device)
                    elif isinstance(batch, (list, tuple)):
                        x = batch[0].to(config.device)
                    else:
                        x = batch.to(config.device)
                    outputs = model(x)

                    # SimpleVAE output format
                    recon_logits = outputs["logits"]
                    mu = outputs["mu"]
                    logvar = outputs["logvar"]
                    z = outputs["z"]

                    loss = loss_fn(recon_logits, x, mu, logvar)
                    val_loss += loss.item()
                    all_z.append(z.cpu())

            val_loss /= len(val_loader)
            all_z = torch.cat(all_z, dim=0)

            # Compute REAL accuracy metrics
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["operation"].to(config.device) if isinstance(batch, dict) else batch[0].to(config.device)
                    outputs = model(x)
                    logits = outputs["logits"]
                    pred = torch.argmax(logits, dim=-1)  # (batch, 9)
                    target = (x + 1).long()  # {-1,0,1} -> {0,1,2}
                    correct += (pred == target).float().sum().item()
                    total += x.size(0) * 9  # 9 positions per sample

            # Use accuracy as both coverage and correlation proxy
            accuracy = correct / total if total > 0 else 0.0
            coverage = accuracy
            correlation = accuracy  # Use same metric for simplicity

            # Track best and early stopping
            if coverage > best_coverage:
                best_coverage = coverage
                best_epoch = epoch
            if correlation > best_correlation:
                best_correlation = correlation
            if val_loss < best_val_loss - config.early_stopping_min_delta:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping check
            if config.enable_early_stopping and patience_counter >= config.early_stopping_patience:
                print(f"[{config.name}] Early stopping at epoch {epoch} (no improvement for {patience_counter} epochs)")
                early_stopped = True
                break

            # Update detector
            metrics = EpochMetrics(
                epoch=epoch,
                train_loss=epoch_loss,
                val_loss=val_loss,
                correlation=correlation,
                coverage=coverage,
            )
            detector.update(metrics)

            # Progress
            if epoch % 10 == 0:
                print(
                    f"[{config.name}] Epoch {epoch}: loss={epoch_loss:.4f}, acc={accuracy:.1%}, val_loss={val_loss:.4f}"
                )

        # ================================================================
        # FINAL RESULTS
        # ================================================================
        # Check for loss spikes
        if len(loss_history) > 1:
            max_ratio = max(loss_history[i] / max(loss_history[i - 1], 1e-8) for i in range(1, len(loss_history)))
            result.max_spike_ratio = max_ratio
            result.had_loss_spike = max_ratio > 2.0

        result.best_coverage = best_coverage
        result.best_correlation = best_correlation
        result.best_val_loss = best_val_loss
        result.best_epoch = best_epoch
        result.final_coverage = coverage
        result.final_correlation = correlation
        result.final_loss = epoch_loss
        result.training_time = time.time() - start_time
        result.early_stopped = early_stopped
        result.early_stop_epoch = epoch if early_stopped else 0
        result.total_epochs_run = epochs_run
        result.completed = True

        stop_info = f" (early stopped at {epoch})" if early_stopped else ""
        print(f"[{config.name}] DONE: coverage={best_coverage:.1%}, correlation={best_correlation:.3f}{stop_info}")

    except Exception as e:
        result.error = str(e)
        result.training_time = time.time() - start_time
        print(f"[{config.name}] ERROR: {e}")

    return result


def run_quick_ablation_test():
    """Quick test to verify ablation training works."""
    print("\n" + "=" * 60)
    print("QUICK ABLATION TEST")
    print("=" * 60)

    # Test baseline
    baseline_config = AblationConfig(
        name="baseline_test",
        epochs=5,
        batch_size=64,
    )

    result = run_ablation_training(baseline_config)
    print(f"\nBaseline result: {result}")

    return result.completed


if __name__ == "__main__":
    run_quick_ablation_test()
