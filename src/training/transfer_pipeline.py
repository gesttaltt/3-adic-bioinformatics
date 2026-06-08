# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Unified Transfer Learning Pipeline.

This module provides a pre-train → fine-tune pipeline that leverages
multi-disease data for improved drug resistance prediction.

Key components:
1. Multi-task pretraining on all diseases
2. Fine-tuning on target disease
3. Few-shot adaptation using MAML
4. Cross-disease transfer evaluation

The pipeline integrates:
- src/models/multi_task_vae.py (shared encoder)
- src/models/maml_vae.py (meta-learning)
- src/models/plm/esm_finetuning.py (PLM fine-tuning)

Usage:
    from src.training.transfer_pipeline import TransferLearningPipeline

    pipeline = TransferLearningPipeline(config)

    # Pre-train on all diseases
    pretrained = pipeline.pretrain(all_disease_data)

    # Fine-tune on target
    finetuned = pipeline.finetune("hiv", hiv_data)

    # Evaluate transfer
    metrics = pipeline.evaluate_transfer("hbv", "hiv")
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class TransferStrategy(Enum):
    """Transfer learning strategies."""

    FULL_FINETUNE = "full"  # Fine-tune all parameters
    FROZEN_ENCODER = "frozen_encoder"  # Freeze encoder, train head
    ADAPTER = "adapter"  # Add adapter layers
    LORA = "lora"  # Low-rank adaptation
    MAML = "maml"  # Model-agnostic meta-learning


@dataclass
class TransferConfig:
    """Configuration for transfer learning pipeline.

    Attributes:
        latent_dim: Latent space dimension
        hidden_dims: Hidden layer dimensions
        pretrain_epochs: Epochs for pretraining
        finetune_epochs: Epochs for fine-tuning
        pretrain_lr: Learning rate for pretraining
        finetune_lr: Learning rate for fine-tuning
        batch_size: Batch size
        strategy: Transfer learning strategy
        maml_inner_lr: MAML inner loop learning rate
        maml_inner_steps: MAML inner loop steps
        adapter_dim: Adapter bottleneck dimension
        lora_rank: LoRA rank
        freeze_layers: Layers to freeze during fine-tuning
        checkpoint_dir: Directory for saving checkpoints
    """

    latent_dim: int = 32
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    pretrain_epochs: int = 100
    finetune_epochs: int = 50
    pretrain_lr: float = 1e-3
    finetune_lr: float = 1e-4
    batch_size: int = 32
    strategy: TransferStrategy = TransferStrategy.FROZEN_ENCODER
    maml_inner_lr: float = 0.01
    maml_inner_steps: int = 5
    adapter_dim: int = 64
    lora_rank: int = 8
    freeze_layers: list[str] = field(default_factory=list)
    checkpoint_dir: Path = Path("checkpoints")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class SharedEncoder(nn.Module):
    """Shared encoder for multi-disease pretraining."""

    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list[int]):
        super().__init__()

        layers = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, h_dim),
                    nn.GELU(),
                    nn.BatchNorm1d(h_dim),
                    nn.Dropout(0.1),
                ]
            )
            in_dim = h_dim

        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)


class DiseaseHead(nn.Module):
    """Disease-specific prediction head."""

    def __init__(self, latent_dim: int, n_outputs: int, hidden_dim: int = 64):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_outputs),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)


class MultiDiseaseModel(nn.Module):
    """Multi-task model with shared encoder and disease-specific heads."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list[int],
        disease_outputs: dict[str, int],
    ):
        """Initialize multi-disease model.

        Args:
            input_dim: Input dimension
            latent_dim: Latent space dimension
            hidden_dims: Hidden layer dimensions
            disease_outputs: Dict mapping disease name to number of outputs
        """
        super().__init__()

        self.encoder = SharedEncoder(input_dim, latent_dim, hidden_dims)
        self.latent_dim = latent_dim

        # Disease-specific heads
        self.heads = nn.ModuleDict(
            {disease: DiseaseHead(latent_dim, n_out) for disease, n_out in disease_outputs.items()}
        )

        # Decoder (shared)
        self.decoder = self._build_decoder(latent_dim, input_dim, hidden_dims)

    def _build_decoder(self, latent_dim: int, output_dim: int, hidden_dims: list[int]) -> nn.Module:
        layers = []
        in_dim = latent_dim

        for h_dim in reversed(hidden_dims):
            layers.extend([nn.Linear(in_dim, h_dim), nn.GELU(), nn.BatchNorm1d(h_dim)])
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_dim))
        return nn.Sequential(*layers)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(
        self,
        x: torch.Tensor,
        disease: str,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor
            disease: Disease name for selecting head

        Returns:
            Dictionary with logits, predictions, mu, logvar, z
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(z)

        predictions = self.heads[disease](z) if disease in self.heads else None

        return {
            "logits": logits,
            "predictions": predictions,
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }

    def get_encoder_params(self):
        """Get encoder parameters."""
        return self.encoder.parameters()

    def get_head_params(self, disease: str):
        """Get disease-specific head parameters."""
        if disease in self.heads:
            return self.heads[disease].parameters()
        return iter([])


class AdapterLayer(nn.Module):
    """Adapter layer for efficient fine-tuning."""

    def __init__(self, input_dim: int, adapter_dim: int):
        super().__init__()

        self.down = nn.Linear(input_dim, adapter_dim)
        self.up = nn.Linear(adapter_dim, input_dim)
        self.act = nn.GELU()

        # Initialize near-identity
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer."""

    def __init__(self, input_dim: int, output_dim: int, rank: int = 8):
        super().__init__()

        self.lora_A = nn.Parameter(torch.randn(input_dim, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, output_dim))
        self.scaling = 1.0 / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.lora_A @ self.lora_B * self.scaling


class TransferLearningPipeline:
    """Unified pre-train → fine-tune pipeline."""

    def __init__(self, config: TransferConfig):
        """Initialize pipeline.

        Args:
            config: Transfer learning configuration
        """
        self.config = config
        self.device = torch.device(config.device)
        self.pretrained_model: MultiDiseaseModel | None = None
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def pretrain(
        self,
        disease_datasets: dict[str, Dataset],
        disease_outputs: dict[str, int],
        input_dim: int,
        callback: Callable | None = None,
    ) -> MultiDiseaseModel:
        """Pre-train on all diseases to learn universal patterns.

        Args:
            disease_datasets: Dict mapping disease name to dataset
            disease_outputs: Dict mapping disease name to number of output drugs
            input_dim: Input dimension
            callback: Optional callback(epoch, losses) for monitoring

        Returns:
            Pre-trained multi-disease model
        """
        # Create model
        model = MultiDiseaseModel(
            input_dim=input_dim,
            latent_dim=self.config.latent_dim,
            hidden_dims=self.config.hidden_dims,
            disease_outputs=disease_outputs,
        ).to(self.device)

        # Create data loaders
        loaders = {
            disease: DataLoader(ds, batch_size=self.config.batch_size, shuffle=True)
            for disease, ds in disease_datasets.items()
        }

        # Optimizer
        optimizer = optim.AdamW(model.parameters(), lr=self.config.pretrain_lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.pretrain_epochs)

        # Training loop
        for epoch in range(self.config.pretrain_epochs):
            model.train()
            epoch_losses = {}

            for disease, loader in loaders.items():
                disease_loss = 0.0
                n_batches = 0

                for batch in loader:
                    x = batch["x"].to(self.device)
                    y = batch.get("y")
                    if y is not None:
                        y = y.to(self.device)

                    optimizer.zero_grad()

                    outputs = model(x, disease)

                    # VAE loss
                    recon_loss = nn.functional.mse_loss(outputs["logits"], x)
                    kl_loss = -0.5 * torch.mean(1 + outputs["logvar"] - outputs["mu"].pow(2) - outputs["logvar"].exp())

                    # Prediction loss (if targets available)
                    if y is not None and outputs["predictions"] is not None:
                        pred_loss = nn.functional.mse_loss(outputs["predictions"].squeeze(), y.squeeze())
                        loss = recon_loss + 0.001 * kl_loss + pred_loss
                    else:
                        loss = recon_loss + 0.001 * kl_loss

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    disease_loss += loss.item()
                    n_batches += 1

                epoch_losses[disease] = disease_loss / max(n_batches, 1)

            scheduler.step()

            if callback:
                callback(epoch, epoch_losses)

            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(model, f"pretrain_epoch_{epoch + 1}.pt")

        self.pretrained_model = model
        self._save_checkpoint(model, "pretrained_final.pt")

        return model

    def finetune(
        self,
        target_disease: str,
        target_dataset: Dataset,
        n_outputs: int,
        input_dim: int | None = None,
        few_shot: bool = False,
        callback: Callable | None = None,
    ) -> nn.Module:
        """Fine-tune on target disease.

        Args:
            target_disease: Name of target disease
            target_dataset: Dataset for target disease
            n_outputs: Number of output predictions
            input_dim: Input dimension (required if no pretrained model)
            few_shot: Whether to use few-shot (MAML) adaptation
            callback: Optional callback for monitoring

        Returns:
            Fine-tuned model
        """
        if self.pretrained_model is None:
            if input_dim is None:
                raise ValueError("input_dim required when no pretrained model available")
            # Create new model
            model = MultiDiseaseModel(
                input_dim=input_dim,
                latent_dim=self.config.latent_dim,
                hidden_dims=self.config.hidden_dims,
                disease_outputs={target_disease: n_outputs},
            ).to(self.device)
        else:
            model = self.pretrained_model
            # Add head for new disease if needed
            if target_disease not in model.heads:
                model.heads[target_disease] = DiseaseHead(model.latent_dim, n_outputs).to(self.device)

        if few_shot and self.config.strategy == TransferStrategy.MAML:
            return self._maml_adapt(model, target_disease, target_dataset, callback)

        # Apply transfer strategy
        model = self._apply_transfer_strategy(model, target_disease)

        # Create data loader
        loader = DataLoader(target_dataset, batch_size=self.config.batch_size, shuffle=True)

        # Get trainable parameters based on strategy
        trainable_params = self._get_trainable_params(model, target_disease)

        optimizer = optim.AdamW(trainable_params, lr=self.config.finetune_lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.finetune_epochs)

        # Fine-tuning loop
        for epoch in range(self.config.finetune_epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in loader:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)

                optimizer.zero_grad()

                outputs = model(x, target_disease)

                # Prediction loss
                pred_loss = nn.functional.mse_loss(outputs["predictions"].squeeze(), y.squeeze())

                # Optional regularization
                if self.config.strategy != TransferStrategy.FULL_FINETUNE:
                    # Smaller weight decay for non-full fine-tuning
                    reg_loss = sum(p.pow(2).sum() for p in trainable_params) * 1e-5
                    loss = pred_loss + reg_loss
                else:
                    loss = pred_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()

            if callback:
                callback(epoch, {"loss": epoch_loss / max(n_batches, 1)})

        # Save fine-tuned model
        self._save_checkpoint(model, f"finetuned_{target_disease}.pt")

        return model

    def _apply_transfer_strategy(self, model: MultiDiseaseModel, target_disease: str) -> MultiDiseaseModel:
        """Apply transfer strategy to model."""
        strategy = self.config.strategy

        if strategy == TransferStrategy.FROZEN_ENCODER:
            # Freeze encoder
            for param in model.encoder.parameters():
                param.requires_grad = False

        elif strategy == TransferStrategy.ADAPTER:
            # Add adapter layers
            self._add_adapters(model)

        elif strategy == TransferStrategy.LORA:
            # Add LoRA layers
            self._add_lora(model)

        return model

    def _add_adapters(self, model: MultiDiseaseModel):
        """Add adapter layers to model."""
        # Add adapters after each layer in encoder
        adapted_layers = []
        for module in model.encoder.encoder:
            adapted_layers.append(module)
            if isinstance(module, nn.Linear):
                adapted_layers.append(AdapterLayer(module.out_features, self.config.adapter_dim))

        model.encoder.encoder = nn.Sequential(*adapted_layers)

        # Freeze original parameters, only train adapters
        for name, param in model.encoder.named_parameters():
            if "adapter" not in name.lower():
                param.requires_grad = False

    def _add_lora(self, model: MultiDiseaseModel):
        """Add LoRA layers to model."""
        # This is a simplified version - production would wrap Linear layers
        model.lora_layers = nn.ModuleList()
        for module in model.encoder.encoder:
            if isinstance(module, nn.Linear):
                lora = LoRALayer(module.in_features, module.out_features, self.config.lora_rank)
                model.lora_layers.append(lora)

        # Freeze base model
        for param in model.encoder.parameters():
            param.requires_grad = False

    def _get_trainable_params(self, model: MultiDiseaseModel, disease: str):
        """Get trainable parameters based on strategy."""
        strategy = self.config.strategy

        if strategy == TransferStrategy.FULL_FINETUNE:
            return model.parameters()

        elif strategy == TransferStrategy.FROZEN_ENCODER:
            return list(model.heads[disease].parameters()) + list(model.decoder.parameters())

        elif strategy == TransferStrategy.ADAPTER:
            params = []
            for name, param in model.named_parameters():
                if "adapter" in name.lower() or disease in name.lower():
                    params.append(param)
            return params

        elif strategy == TransferStrategy.LORA:
            return list(model.lora_layers.parameters()) + list(model.heads[disease].parameters())

        return model.parameters()

    def _maml_adapt(
        self,
        model: MultiDiseaseModel,
        disease: str,
        dataset: Dataset,
        callback: Callable | None,
    ) -> nn.Module:
        """MAML-style few-shot adaptation."""
        import copy

        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        for step in range(self.config.maml_inner_steps):
            # Clone model for inner loop
            model_copy = copy.deepcopy(model)
            optimizer = optim.SGD(model_copy.parameters(), lr=self.config.maml_inner_lr)

            for batch in loader:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)

                optimizer.zero_grad()
                outputs = model_copy(x, disease)
                loss = nn.functional.mse_loss(outputs["predictions"].squeeze(), y.squeeze())
                loss.backward()
                optimizer.step()

            if callback:
                callback(step, {"maml_loss": loss.item()})

        return model_copy

    def evaluate_transfer(
        self,
        source_disease: str,
        target_disease: str,
        target_dataset: Dataset,
    ) -> dict[str, float]:
        """Evaluate transfer from source to target disease.

        Args:
            source_disease: Disease used for pretraining
            target_disease: Target disease for evaluation
            target_dataset: Test dataset for target disease

        Returns:
            Dictionary of evaluation metrics
        """
        if self.pretrained_model is None:
            return {"error": "No pretrained model available"}

        model = self.pretrained_model
        model.eval()

        loader = DataLoader(target_dataset, batch_size=self.config.batch_size)

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)

                # Use source disease head (zero-shot transfer)
                if source_disease in model.heads:
                    outputs = model(x, source_disease)
                    all_preds.append(outputs["predictions"].cpu())
                    all_targets.append(y.cpu())

        if not all_preds:
            return {"error": "No predictions made"}

        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)

        # Compute metrics
        mse = nn.functional.mse_loss(preds.squeeze(), targets.squeeze()).item()
        from scipy.stats import spearmanr

        corr, p_value = spearmanr(preds.numpy().flatten(), targets.numpy().flatten())

        return {
            "mse": mse,
            "spearman_correlation": corr,
            "p_value": p_value,
            "source": source_disease,
            "target": target_disease,
        }

    def _save_checkpoint(self, model: nn.Module, filename: str):
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": self.config,
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> nn.Module:
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        # Would need to reconstruct model architecture here
        return checkpoint


__all__ = [
    "TransferLearningPipeline",
    "TransferConfig",
    "TransferStrategy",
    "MultiDiseaseModel",
    "SharedEncoder",
    "DiseaseHead",
    "AdapterLayer",
    "LoRALayer",
]
