# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Self-Supervised Pre-training for Biological Sequence VAEs.

This module provides comprehensive self-supervised pre-training for
biological sequences, combining multiple objectives:

1. Masked Sequence Modeling (MSM): Reconstruct masked positions
2. Contrastive Learning (BYOL/SimCLR): Learn invariances to augmentations
3. VAE Reconstruction: Learn latent representations
4. Sequence Prediction: Next/Previous token prediction
5. Mutation Impact Prediction: Predict effect of mutations

These objectives enable learning rich representations from unlabeled
sequence data before fine-tuning on drug resistance prediction.

Example:
    from src.training.self_supervised import SelfSupervisedPretrainer

    pretrainer = SelfSupervisedPretrainer(config)

    # Pre-train on unlabeled sequences
    pretrained_encoder = pretrainer.pretrain(sequences_dataset)

    # Fine-tune on labeled data
    model = pretrainer.create_downstream_model(pretrained_encoder)
    model = fine_tune(model, labeled_dataset)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class PretrainingObjective(Enum):
    """Available pre-training objectives."""

    MSM = "masked_sequence_modeling"  # Masked token prediction
    CONTRASTIVE = "contrastive"  # BYOL/SimCLR style
    VAE = "vae"  # Reconstruction + KL
    NEXT_TOKEN = "next_token"  # Next token prediction
    MUTATION = "mutation"  # Mutation impact prediction


@dataclass
class SelfSupervisedConfig:
    """Configuration for self-supervised pre-training.

    Attributes:
        input_dim: Input dimension per position
        seq_length: Sequence length
        hidden_dim: Hidden dimension
        latent_dim: Latent space dimension
        n_layers: Number of encoder layers
        n_heads: Number of attention heads
        dropout: Dropout rate
        mask_ratio: Ratio of positions to mask for MSM
        objectives: Which objectives to use
        objective_weights: Weights for each objective
        temperature: Temperature for contrastive loss
        momentum: Momentum for target encoder (BYOL)
        pretrain_epochs: Number of pre-training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        warmup_steps: Warmup steps for scheduler
        checkpoint_dir: Directory for checkpoints
        device: Device to use
    """

    input_dim: int = 21  # 20 amino acids + gap
    seq_length: int = 99
    hidden_dim: int = 256
    latent_dim: int = 64
    n_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.1
    mask_ratio: float = 0.15
    objectives: list[PretrainingObjective] = field(
        default_factory=lambda: [
            PretrainingObjective.MSM,
            PretrainingObjective.CONTRASTIVE,
            PretrainingObjective.VAE,
        ]
    )
    objective_weights: dict[str, float] = field(
        default_factory=lambda: {
            "msm": 1.0,
            "contrastive": 0.5,
            "vae_recon": 1.0,
            "vae_kl": 0.001,
            "next_token": 0.5,
            "mutation": 0.5,
        }
    )
    temperature: float = 0.07
    momentum: float = 0.996
    pretrain_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    checkpoint_dir: Path = Path("checkpoints/pretrain")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class SequenceEncoder(nn.Module):
    """Transformer-based sequence encoder for pre-training."""

    def __init__(self, config: SelfSupervisedConfig):
        super().__init__()

        self.config = config

        # Input embedding
        self.embedding = nn.Linear(config.input_dim, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.seq_length, config.hidden_dim)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        # Output projections
        self.to_latent = nn.Linear(config.hidden_dim, config.latent_dim * 2)  # mu, logvar
        self.layer_norm = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        return_sequence: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Encode sequence.

        Args:
            x: Input tensor (batch, seq_len, input_dim) or (batch, seq_len * input_dim)
            return_sequence: Whether to return full sequence embedding

        Returns:
            Dictionary with mu, logvar, z, and optionally sequence embeddings
        """
        batch_size = x.size(0)

        # Reshape if flattened
        if x.dim() == 2:
            x = x.view(batch_size, self.config.seq_length, self.config.input_dim)

        # Embed
        h = self.embedding(x)

        # Add position embeddings
        positions = torch.arange(x.size(1), device=x.device)
        h = h + self.position_embedding(positions)

        # Transformer encoding
        h = self.transformer(h)
        h = self.layer_norm(h)

        # Pool to get sequence representation (mean pooling)
        pooled = h.mean(dim=1)

        # Project to latent
        latent = self.to_latent(pooled)
        mu, logvar = latent.chunk(2, dim=-1)

        # Reparameterize
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu

        outputs = {"mu": mu, "logvar": logvar, "z": z}

        if return_sequence:
            outputs["sequence_embeddings"] = h

        return outputs


class MaskedSequenceModeling(nn.Module):
    """Masked Sequence Modeling head for pre-training."""

    def __init__(self, config: SelfSupervisedConfig):
        super().__init__()

        self.config = config

        # Decoder for predicting masked tokens
        self.decoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.input_dim),
        )

    def forward(
        self,
        sequence_embeddings: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict masked tokens.

        Args:
            sequence_embeddings: Encoder output (batch, seq_len, hidden_dim)
            mask: Boolean mask (batch, seq_len), True for masked positions

        Returns:
            Predictions for masked positions
        """
        # Only decode masked positions
        masked_embeddings = sequence_embeddings[mask]
        predictions = self.decoder(masked_embeddings)
        return predictions

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSM loss."""
        return F.cross_entropy(predictions, targets)


class SequenceDecoder(nn.Module):
    """Decoder for VAE reconstruction."""

    def __init__(self, config: SelfSupervisedConfig):
        super().__init__()

        self.config = config

        # Projection from latent to hidden
        self.from_latent = nn.Linear(config.latent_dim, config.hidden_dim * config.seq_length)

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.input_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to sequence.

        Args:
            z: Latent vector (batch, latent_dim)

        Returns:
            Reconstruction logits (batch, seq_len, input_dim)
        """
        batch_size = z.size(0)

        # Project and reshape
        h = self.from_latent(z)
        h = h.view(batch_size, self.config.seq_length, self.config.hidden_dim)

        # Decode each position
        logits = self.decoder(h)
        return logits


class ContrastiveHead(nn.Module):
    """Contrastive learning projection head."""

    def __init__(self, config: SelfSupervisedConfig):
        super().__init__()

        self.projector = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )

        # Predictor (for BYOL asymmetry)
        self.predictor = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project latent for contrastive learning.

        Args:
            z: Latent vector (batch, latent_dim)

        Returns:
            Tuple of (projection, prediction)
        """
        projection = self.projector(z)
        prediction = self.predictor(projection)
        return projection, prediction


class MutationHead(nn.Module):
    """Head for mutation impact prediction."""

    def __init__(self, config: SelfSupervisedConfig):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(config.latent_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(
        self,
        z_original: torch.Tensor,
        z_mutant: torch.Tensor,
    ) -> torch.Tensor:
        """Predict mutation impact.

        Args:
            z_original: Latent for original sequence
            z_mutant: Latent for mutant sequence

        Returns:
            Predicted impact score
        """
        combined = torch.cat([z_original, z_mutant], dim=-1)
        return self.head(combined).squeeze(-1)


class SequenceAugmenter:
    """Augmentations for contrastive learning."""

    def __init__(
        self,
        mask_prob: float = 0.15,
        replace_prob: float = 0.1,
        n_classes: int = 21,
    ):
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob
        self.n_classes = n_classes

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply augmentation.

        Args:
            x: Input tensor (batch, seq_len, input_dim) or (batch, seq_len)

        Returns:
            Augmented tensor
        """
        x = x.clone()

        # Random masking (set to zero)
        mask = torch.rand_like(x[..., 0] if x.dim() == 3 else x.float()) < self.mask_prob
        if x.dim() == 3:
            x[mask] = 0
        else:
            x[mask] = 0

        # Random replacement
        replace_mask = torch.rand_like(x[..., 0] if x.dim() == 3 else x.float()) < self.replace_prob
        if x.dim() == 3:
            n_replace = replace_mask.sum().item()
            if n_replace > 0:
                input_dim = x.size(-1)
                n_classes = min(self.n_classes, input_dim)
                random_tokens = torch.zeros(n_replace, input_dim, device=x.device, dtype=x.dtype)
                random_idx = torch.randint(0, n_classes, (n_replace,), device=x.device)
                random_tokens.scatter_(1, random_idx.unsqueeze(1), 1)
                x[replace_mask] = random_tokens
        else:
            n_replace = replace_mask.sum().item()
            if n_replace > 0:
                x[replace_mask] = torch.randint(0, self.n_classes, (n_replace,), device=x.device).float()

        return x


class SelfSupervisedModel(nn.Module):
    """Complete self-supervised pre-training model."""

    def __init__(self, config: SelfSupervisedConfig):
        super().__init__()

        self.config = config

        # Encoder
        self.encoder = SequenceEncoder(config)

        # Task-specific heads
        if PretrainingObjective.MSM in config.objectives:
            self.msm_head = MaskedSequenceModeling(config)

        if PretrainingObjective.VAE in config.objectives:
            self.decoder = SequenceDecoder(config)

        if PretrainingObjective.CONTRASTIVE in config.objectives:
            self.contrastive_head = ContrastiveHead(config)
            # Target encoder (momentum updated)
            self.target_encoder = SequenceEncoder(config)
            self.target_contrastive = ContrastiveHead(config)
            # Initialize target with same weights
            self._copy_params()

        if PretrainingObjective.MUTATION in config.objectives:
            self.mutation_head = MutationHead(config)

        self.augmenter = SequenceAugmenter(mask_prob=config.mask_ratio)

    def _copy_params(self):
        """Copy parameters from online to target encoder."""
        for p_online, p_target in zip(
            self.encoder.parameters(),
            self.target_encoder.parameters(),
            strict=False,
        ):
            p_target.data.copy_(p_online.data)
            p_target.requires_grad = False

        for p_online, p_target in zip(
            self.contrastive_head.parameters(),
            self.target_contrastive.parameters(),
            strict=False,
        ):
            p_target.data.copy_(p_online.data)
            p_target.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self):
        """Momentum update of target network (BYOL-style)."""
        m = self.config.momentum

        for p_online, p_target in zip(
            self.encoder.parameters(),
            self.target_encoder.parameters(),
            strict=False,
        ):
            p_target.data = m * p_target.data + (1 - m) * p_online.data

        for p_online, p_target in zip(
            self.contrastive_head.parameters(),
            self.target_contrastive.parameters(),
            strict=False,
        ):
            p_target.data = m * p_target.data + (1 - m) * p_online.data

    def forward(
        self,
        x: torch.Tensor,
        compute_contrastive: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for all objectives.

        Args:
            x: Input sequences (batch, seq_len, input_dim) or (batch, seq_len * input_dim)
            compute_contrastive: Whether to compute contrastive objectives

        Returns:
            Dictionary with all outputs
        """
        outputs = {}

        # Encode original sequence
        encoder_out = self.encoder(x, return_sequence=True)
        outputs.update(encoder_out)

        # VAE reconstruction
        if hasattr(self, "decoder"):
            logits = self.decoder(encoder_out["z"])
            outputs["reconstruction"] = logits

        # Contrastive (BYOL-style)
        if hasattr(self, "contrastive_head") and compute_contrastive:
            # Online projection
            proj_online, pred_online = self.contrastive_head(encoder_out["z"])
            outputs["proj_online"] = proj_online
            outputs["pred_online"] = pred_online

            # Create augmented view
            x_aug = self.augmenter(x)
            encoder_aug = self.encoder(x_aug, return_sequence=False)
            proj_aug, _ = self.contrastive_head(encoder_aug["z"])
            outputs["proj_aug"] = proj_aug

            # Target projection (no gradient)
            with torch.no_grad():
                target_out = self.target_encoder(x, return_sequence=False)
                proj_target, _ = self.target_contrastive(target_out["z"])
                outputs["proj_target"] = proj_target

                target_aug = self.target_encoder(x_aug, return_sequence=False)
                proj_target_aug, _ = self.target_contrastive(target_aug["z"])
                outputs["proj_target_aug"] = proj_target_aug

        return outputs

    def compute_losses(
        self,
        x: torch.Tensor,
        outputs: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute all pre-training losses.

        Args:
            x: Original input
            outputs: Pre-computed outputs (if None, runs forward pass)

        Returns:
            Dictionary of losses
        """
        if outputs is None:
            outputs = self.forward(x)

        losses = {}
        weights = self.config.objective_weights

        # VAE reconstruction loss
        if "reconstruction" in outputs:
            batch_size = x.size(0)
            x_flat = x.view(batch_size, -1)
            recon_flat = outputs["reconstruction"].view(batch_size, -1)

            losses["recon"] = F.mse_loss(recon_flat, x_flat) * weights.get("vae_recon", 1.0)

            # KL divergence
            kl = -0.5 * torch.mean(1 + outputs["logvar"] - outputs["mu"].pow(2) - outputs["logvar"].exp())
            losses["kl"] = kl * weights.get("vae_kl", 0.001)

        # Contrastive loss (BYOL)
        if "pred_online" in outputs:
            # Loss 1: predict target from online
            loss1 = (
                1
                - F.cosine_similarity(
                    outputs["pred_online"],
                    outputs["proj_target"].detach(),
                    dim=-1,
                ).mean()
            )

            # Loss 2: predict target from augmented online
            pred_aug, _ = self.contrastive_head(self.encoder(self.augmenter(x), return_sequence=False)["z"])
            loss2 = (
                1
                - F.cosine_similarity(
                    pred_aug,
                    outputs["proj_target_aug"].detach(),
                    dim=-1,
                ).mean()
            )

            losses["contrastive"] = (loss1 + loss2) / 2 * weights.get("contrastive", 0.5)

        # Total loss
        losses["total"] = sum(v for k, v in losses.items() if k != "total")

        return losses

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get sequence embeddings (for downstream tasks).

        Args:
            x: Input sequences

        Returns:
            Latent embeddings
        """
        self.eval()
        with torch.no_grad():
            outputs = self.encoder(x)
            return outputs["mu"]


class SelfSupervisedPretrainer:
    """Pre-training pipeline for self-supervised learning."""

    def __init__(self, config: SelfSupervisedConfig):
        """Initialize pretrainer.

        Args:
            config: Pre-training configuration
        """
        self.config = config
        self.device = torch.device(config.device)
        self.model: SelfSupervisedModel | None = None
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def pretrain(
        self,
        dataset: Dataset,
        callback: Callable | None = None,
    ) -> SequenceEncoder:
        """Pre-train on unlabeled sequence data.

        Args:
            dataset: Dataset of sequences (each item has "x" key)
            callback: Optional callback(epoch, losses) for monitoring

        Returns:
            Pre-trained encoder
        """
        # Create model
        self.model = SelfSupervisedModel(self.config).to(self.device)

        # Data loader
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

        # Optimizer with weight decay
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )

        # Scheduler
        total_steps = len(loader) * self.config.pretrain_epochs
        # Clamp pct_start to valid range [0, 1]
        pct_start = min(max(self.config.warmup_steps / max(total_steps, 1), 0.0), 0.3)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.learning_rate,
            total_steps=max(total_steps, 1),
            pct_start=pct_start,
        )

        # Training loop
        for epoch in range(self.config.pretrain_epochs):
            self.model.train()
            epoch_losses = {"total": 0.0}

            for batch in loader:
                x = batch["x"].to(self.device)

                optimizer.zero_grad()

                outputs = self.model(x)
                losses = self.model.compute_losses(x, outputs)

                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                # Momentum update for BYOL
                if hasattr(self.model, "target_encoder"):
                    self.model._momentum_update()

                for k, v in losses.items():
                    if k not in epoch_losses:
                        epoch_losses[k] = 0.0
                    epoch_losses[k] += v.item()

            # Average losses
            n_batches = len(loader)
            for k in epoch_losses:
                epoch_losses[k] /= n_batches

            if callback:
                callback(epoch, epoch_losses)

            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch + 1)

        # Final checkpoint
        self._save_checkpoint("final")

        return self.model.encoder

    def _save_checkpoint(self, identifier: int | str):
        """Save model checkpoint."""
        path = self.checkpoint_dir / f"pretrain_{identifier}.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "encoder_state_dict": self.model.encoder.state_dict(),
                "config": self.config,
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> SequenceEncoder:
        """Load pre-trained encoder from checkpoint.

        Args:
            path: Path to checkpoint

        Returns:
            Pre-trained encoder
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Recreate encoder
        encoder = SequenceEncoder(self.config).to(self.device)
        encoder.load_state_dict(checkpoint["encoder_state_dict"])

        return encoder

    def create_downstream_model(
        self,
        encoder: SequenceEncoder | None = None,
        n_outputs: int = 1,
        freeze_encoder: bool = True,
    ) -> nn.Module:
        """Create model for downstream fine-tuning.

        Args:
            encoder: Pre-trained encoder (uses self.model.encoder if None)
            n_outputs: Number of output predictions
            freeze_encoder: Whether to freeze encoder weights

        Returns:
            Model ready for fine-tuning
        """
        if encoder is None:
            if self.model is None:
                raise ValueError("No pretrained model available")
            encoder = self.model.encoder

        class DownstreamModel(nn.Module):
            def __init__(self, encoder, config, n_outputs, freeze):
                super().__init__()
                self.encoder = encoder
                self.head = nn.Sequential(
                    nn.Linear(config.latent_dim, config.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.hidden_dim, n_outputs),
                )

                if freeze:
                    for param in self.encoder.parameters():
                        param.requires_grad = False

            def forward(self, x):
                enc_out = self.encoder(x)
                predictions = self.head(enc_out["z"])
                return {
                    "predictions": predictions,
                    "z": enc_out["z"],
                    "mu": enc_out["mu"],
                    "logvar": enc_out["logvar"],
                }

        return DownstreamModel(encoder, self.config, n_outputs, freeze_encoder)

    def evaluate_representations(
        self,
        dataset: Dataset,
        method: str = "linear_probe",
    ) -> dict[str, float]:
        """Evaluate quality of learned representations.

        Args:
            dataset: Labeled dataset for evaluation
            method: Evaluation method ('linear_probe' or 'knn')

        Returns:
            Evaluation metrics
        """
        if self.model is None:
            return {"error": "No pretrained model available"}

        self.model.eval()

        # Extract embeddings
        loader = DataLoader(dataset, batch_size=self.config.batch_size)
        embeddings = []
        labels = []

        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(self.device)
                y = batch.get("y")

                emb = self.model.get_embeddings(x)
                embeddings.append(emb.cpu())

                if y is not None:
                    labels.append(y.cpu())

        embeddings = torch.cat(embeddings)

        if not labels:
            return {"embeddings_shape": list(embeddings.shape)}

        labels = torch.cat(labels)

        if method == "linear_probe":
            return self._linear_probe_eval(embeddings, labels)
        elif method == "knn":
            return self._knn_eval(embeddings, labels)

        return {}

    def _linear_probe_eval(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, float]:
        """Linear probe evaluation."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split

        X = embeddings.numpy()
        y = labels.numpy()

        # Handle continuous labels
        if y.dtype == torch.float32 or y.dtype == torch.float64:
            # Regression
            from sklearn.linear_model import Ridge

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            clf = Ridge()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            mse = ((y_pred - y_test) ** 2).mean()
            from scipy.stats import spearmanr

            corr, _ = spearmanr(y_pred.flatten(), y_test.flatten())

            return {"mse": float(mse), "spearman": float(corr)}
        else:
            # Classification
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            return {"accuracy": accuracy_score(y_test, y_pred)}

    def _knn_eval(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        k: int = 5,
    ) -> dict[str, float]:
        """K-NN evaluation."""
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KNeighborsClassifier

        X = embeddings.numpy()
        y = labels.numpy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        return {"knn_accuracy": accuracy_score(y_test, y_pred)}


__all__ = [
    "SelfSupervisedPretrainer",
    "SelfSupervisedConfig",
    "SelfSupervisedModel",
    "SequenceEncoder",
    "SequenceDecoder",
    "ContrastiveHead",
    "MaskedSequenceModeling",
    "MutationHead",
    "SequenceAugmenter",
    "PretrainingObjective",
]
