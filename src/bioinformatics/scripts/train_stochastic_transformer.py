#!/usr/bin/env python3
"""Train Stochastic Transformer using VAE+MLP Refiner embeddings.

Phase 2: Stochastic Transformer
===============================
Uses VAE-ProTherm + MLP Refiner embeddings as inputs for a transformer.
The VAE provides a "fuzzy" topological map, and the transformer learns
to navigate this map for precise DDG prediction.

Key insight: The VAE latent space provides a continuous representation
where similar mutations cluster together. The transformer can learn
attention patterns over this space to discover non-evident relationships.

Architecture:
    Raw Features -> VAE-ProTherm -> MLP Refiner -> Refined Embeddings
    Refined Embeddings -> Transformer -> DDG Prediction

Usage:
    python src/bioinformatics/scripts/train_stochastic_transformer.py
    python src/bioinformatics/scripts/train_stochastic_transformer.py --quick
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

from src.bioinformatics.data.protherm_loader import ProThermLoader
from src.bioinformatics.models.ddg_mlp_refiner import DDGMLPRefiner, RefinerConfig
from src.bioinformatics.models.ddg_vae import DDGVAE
from src.bioinformatics.training.deterministic import set_deterministic_mode


@dataclass
class StochasticTransformerConfig:
    """Configuration for Stochastic Transformer."""

    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 256
    dropout: float = 0.1
    activation: str = "gelu"
    use_residual_from_refiner: bool = True  # Use MLP refiner prediction as residual
    use_multi_head_output: bool = True  # Multiple prediction heads


class StochasticTransformer(nn.Module):
    """Transformer that processes VAE+MLP Refiner embeddings.

    The "stochastic" aspect comes from:
    1. VAE sampling during training (reparameterization trick)
    2. Monte Carlo dropout during inference
    3. Multi-head predictions with learned combination weights

    Architecture:
    - Input: Refined embeddings from VAE+MLP pipeline
    - Positional encoding for embedding dimensions
    - Transformer encoder with CLS token
    - Multi-head output (uncertainty quantification)
    """

    def __init__(
        self,
        config: StochasticTransformerConfig,
        embedding_dim: int,
        n_monte_carlo_samples: int = 5,
    ):
        super().__init__()
        self.config = config
        self.embedding_dim = embedding_dim
        self.n_monte_carlo_samples = n_monte_carlo_samples

        # Input projection (embedding dim -> sequence of tokens)
        self.input_proj = nn.Linear(1, config.d_model)

        # Positional encoding
        self.pos_enc = nn.Parameter(torch.randn(1, embedding_dim, config.d_model) * 0.02)

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

        # Multi-head output
        if config.use_multi_head_output:
            self.heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(config.d_model, config.d_model // 2),
                        nn.SiLU(),
                        nn.Dropout(config.dropout),
                        nn.Linear(config.d_model // 2, 1),
                    )
                    for _ in range(3)  # 3 prediction heads
                ]
            )
            # Learned combination weights
            self.head_weights = nn.Parameter(torch.ones(3) / 3)
        else:
            self.head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.SiLU(),
                nn.LayerNorm(config.d_model),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model, 1),
            )

        # Residual weight for refiner prediction
        if config.use_residual_from_refiner:
            self.residual_weight = nn.Parameter(torch.tensor(0.5))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        embedding: torch.Tensor,
        refiner_pred: torch.Tensor | None = None,
        return_uncertainty: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            embedding: VAE+MLP Refiner embeddings (batch, embedding_dim)
            refiner_pred: Optional MLP Refiner prediction for residual
            return_uncertainty: Return prediction uncertainty

        Returns:
            Dictionary with predictions and optional uncertainty
        """
        batch_size = embedding.shape[0]

        # Treat each embedding dimension as a token
        x = embedding.unsqueeze(-1)  # (batch, embedding_dim, 1)
        x = self.input_proj(x)  # (batch, embedding_dim, d_model)
        x = x + self.pos_enc

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Transform
        x = self.transformer(x)

        # Get CLS output
        cls_out = x[:, 0]  # (batch, d_model)

        # Multi-head prediction
        if self.config.use_multi_head_output:
            head_preds = [head(cls_out) for head in self.heads]
            head_preds = torch.cat(head_preds, dim=-1)  # (batch, n_heads)

            # Weighted combination
            weights = F.softmax(self.head_weights, dim=0)
            transformer_pred = (head_preds * weights).sum(dim=-1, keepdim=True)

            # Uncertainty from head disagreement
            head_uncertainty = head_preds.std(dim=-1, keepdim=True)
        else:
            transformer_pred = self.head(cls_out)
            head_preds = transformer_pred
            head_uncertainty = torch.zeros_like(transformer_pred)

        # Residual from refiner
        if self.config.use_residual_from_refiner and refiner_pred is not None:
            if refiner_pred.dim() == 1:
                refiner_pred = refiner_pred.unsqueeze(-1)
            weight = torch.sigmoid(self.residual_weight)
            ddg_pred = refiner_pred + weight * transformer_pred
        else:
            ddg_pred = transformer_pred

        result = {
            "ddg_pred": ddg_pred,
            "transformer_pred": transformer_pred,
            "head_preds": head_preds,
            "cls_embedding": cls_out,
        }

        if return_uncertainty:
            result["uncertainty"] = head_uncertainty

        return result

    def forward_stochastic(
        self,
        embedding: torch.Tensor,
        refiner_pred: torch.Tensor | None = None,
        n_samples: int = 10,
    ) -> dict[str, torch.Tensor]:
        """Monte Carlo forward pass for uncertainty estimation.

        Runs multiple forward passes with dropout enabled to estimate
        prediction uncertainty.

        Args:
            embedding: Input embeddings
            refiner_pred: Optional refiner predictions
            n_samples: Number of MC samples

        Returns:
            Dictionary with mean prediction and uncertainty
        """
        self.train()  # Enable dropout
        preds = []

        for _ in range(n_samples):
            output = self.forward(embedding, refiner_pred)
            preds.append(output["ddg_pred"])

        preds = torch.stack(preds, dim=0)  # (n_samples, batch, 1)

        mean_pred = preds.mean(dim=0)
        std_pred = preds.std(dim=0)

        return {
            "ddg_pred": mean_pred,
            "ddg_std": std_pred,
            "mc_samples": preds,
        }

    def loss(
        self,
        embedding: torch.Tensor,
        y: torch.Tensor,
        refiner_pred: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> dict[str, torch.Tensor]:
        """Compute loss with optional uncertainty calibration.

        Args:
            embedding: Input embeddings
            y: Target DDG values
            refiner_pred: Optional refiner predictions
            reduction: Loss reduction method

        Returns:
            Dictionary with loss components
        """
        output = self.forward(embedding, refiner_pred, return_uncertainty=True)
        ddg_pred = output["ddg_pred"]
        uncertainty = output["uncertainty"]

        # Ensure y has correct shape
        if y.dim() == 1:
            y = y.unsqueeze(-1)

        # MSE loss
        mse_loss = F.mse_loss(ddg_pred, y, reduction=reduction)

        # Uncertainty calibration loss (optional)
        # Encourage uncertainty to correlate with actual error
        if uncertainty.sum() > 0:
            error = (ddg_pred - y).abs().detach()
            calibration_loss = F.mse_loss(uncertainty, error, reduction=reduction)
            total_loss = mse_loss + 0.1 * calibration_loss
        else:
            calibration_loss = torch.tensor(0.0)
            total_loss = mse_loss

        return {
            "loss": total_loss,
            "mse_loss": mse_loss,
            "calibration_loss": calibration_loss,
        }


class RefinerEmbeddingDataset(Dataset):
    """Dataset of VAE+MLP Refiner embeddings."""

    def __init__(
        self,
        vae: DDGVAE,
        refiner: DDGMLPRefiner,
        base_dataset: Dataset,
        device: str = "cuda",
    ):
        """Extract embeddings from VAE+MLP Refiner pipeline.

        Args:
            vae: Trained VAE model
            refiner: Trained MLP Refiner
            base_dataset: Dataset with (features, labels)
            device: Device for inference
        """
        self.labels = []
        self.embeddings = []
        self.vae_preds = []
        self.refiner_preds = []

        vae = vae.to(device)
        refiner = refiner.to(device)
        vae.eval()
        refiner.eval()

        with torch.no_grad():
            for i in range(len(base_dataset)):
                x, y = base_dataset[i]
                x = x.unsqueeze(0).to(device)

                # VAE forward pass
                vae_output = vae(x)
                mu = vae_output["mu"]
                vae_pred = vae_output["ddg_pred"]

                # MLP Refiner forward pass
                refiner_output = refiner(mu, vae_pred)
                refiner_pred = refiner_output["ddg_pred"]

                # Store embeddings and predictions
                self.embeddings.append(mu.cpu().squeeze(0))
                self.vae_preds.append(vae_pred.cpu().squeeze())
                self.refiner_preds.append(refiner_pred.cpu().squeeze())
                self.labels.append(y)

        self.embeddings = torch.stack(self.embeddings)
        self.vae_preds = torch.stack(self.vae_preds)
        self.refiner_preds = torch.stack(self.refiner_preds)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "embedding": self.embeddings[idx],
            "vae_pred": self.vae_preds[idx],
            "refiner_pred": self.refiner_preds[idx],
            "label": self.labels[idx],
        }

    @property
    def embedding_dim(self):
        return self.embeddings.shape[1]


def compute_qa_metrics(
    model: StochasticTransformer,
    dataset: Dataset,
    device: str = "cuda",
    use_mc: bool = False,
) -> dict[str, float]:
    """Compute QA metrics with optional MC dropout."""
    model.eval() if not use_mc else model.train()
    model.to(device)

    all_preds = []
    all_labels = []
    all_stds = []

    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    with torch.no_grad() if not use_mc else torch.enable_grad():
        for batch in loader:
            embedding = batch["embedding"].to(device)
            refiner_pred = batch["refiner_pred"].to(device)
            y = batch["label"]

            if use_mc:
                with torch.no_grad():
                    output = model.forward_stochastic(embedding, refiner_pred, n_samples=10)
                    pred = output["ddg_pred"].squeeze(-1)
                    std = output["ddg_std"].squeeze(-1)
                    all_stds.extend(std.cpu().numpy())
            else:
                output = model(embedding, refiner_pred)
                pred = output["ddg_pred"].squeeze(-1)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

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

    if use_mc and all_stds:
        all_stds = np.array(all_stds)
        errors = np.abs(all_preds - all_labels)
        # Calibration: correlation between uncertainty and error
        unc_error_corr = np.corrcoef(all_stds, errors)[0, 1]
        result["uncertainty_calibration"] = float(unc_error_corr)
        result["mean_uncertainty"] = float(np.mean(all_stds))

    return result


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

    if "uncertainty_calibration" in metrics:
        print(f"  Uncertainty calibration: {metrics['uncertainty_calibration']:.4f}")
        print(f"  Mean uncertainty: {metrics['mean_uncertainty']:.4f}")

    quality = (
        "EXCELLENT"
        if metrics["spearman"] > 0.8
        else "VERY GOOD"
        if metrics["spearman"] > 0.7
        else "GOOD"
        if metrics["spearman"] > 0.5
        else "MODERATE"
    )
    print(f"\n  Quality:        {quality}")
    print(f"{'=' * 60}")


def train_stochastic_transformer(
    train_dataset: RefinerEmbeddingDataset,
    val_dataset: Dataset,
    output_dir: Path,
    config: StochasticTransformerConfig | None = None,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-4,
    patience: int = 30,
    device: str = "cuda",
    verbose: bool = True,
) -> tuple[StochasticTransformer, dict[str, Any]]:
    """Train Stochastic Transformer on VAE+MLP Refiner embeddings."""

    if config is None:
        config = StochasticTransformerConfig()

    embedding_dim = train_dataset.embedding_dim
    output_dir.mkdir(parents=True, exist_ok=True)

    model = StochasticTransformer(config, embedding_dim).to(device)
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
    print("Training: Stochastic Transformer")
    print(f"{'=' * 60}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples:   {len(val_dataset)}")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  d_model:       {config.d_model}")
    print(f"  n_layers:      {config.n_layers}")
    print(f"  Multi-head:    {config.use_multi_head_output}")
    print(f"  Residual:      {config.use_residual_from_refiner}")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            embedding = batch["embedding"].to(device)
            refiner_pred = batch["refiner_pred"].to(device)
            y = batch["label"].to(device)

            optimizer.zero_grad()
            loss_dict = model.loss(embedding, y, refiner_pred)
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
                embedding = batch["embedding"].to(device)
                refiner_pred = batch["refiner_pred"].to(device)
                y = batch["label"].to(device)

                output = model(embedding, refiner_pred)
                pred = output["ddg_pred"].squeeze(-1)

                loss = F.mse_loss(pred, y)
                val_loss += loss.item()

                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        val_loss /= len(val_loader)
        val_spearman = spearmanr(all_preds, all_labels)[0]
        val_pearson = pearsonr(all_preds, all_labels)[0]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_spearman"].append(float(val_spearman))
        history["val_pearson"].append(float(val_pearson))

        if verbose and epoch % 10 == 0:
            res_weight = torch.sigmoid(model.residual_weight).item() if hasattr(model, "residual_weight") else 0
            print(
                f"  Epoch {epoch:3d}: loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"ρ={val_spearman:.4f} res_w={res_weight:.3f}"
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

    # Load best model
    ckpt = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    print("\n  Training complete!")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best Spearman: {best_spearman:.4f}")

    return model, history


class SubsetWrapper(Dataset):
    """Wrapper for random_split subsets to preserve embedding_dim."""

    def __init__(self, subset, embedding_dim):
        self.subset = subset
        self.embedding_dim = embedding_dim

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        return self.subset[idx]


def main():
    parser = argparse.ArgumentParser(description="Train Stochastic Transformer")
    parser.add_argument("--quick", action="store_true", help="Quick test run")
    parser.add_argument("--device", default="cuda", help="Device")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    set_deterministic_mode(42)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs/stochastic_transformer_{timestamp}")

    epochs = 30 if args.quick else 150
    patience = 10 if args.quick else 40

    print("=" * 70)
    print("STOCHASTIC TRANSFORMER TRAINING")
    print("VAE-ProTherm + MLP Refiner Embeddings -> Transformer -> DDG")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {epochs}, Patience: {patience}")

    # =========================================================
    # Load VAE-ProTherm and MLP Refiner
    # =========================================================
    print("\n[1] Loading VAE-ProTherm...")

    vae = DDGVAE.create_protherm_variant(use_hyperbolic=False)
    vae_path = Path("outputs/ddg_vae_training_20260129_212316/vae_protherm/best.pt")

    if vae_path.exists():
        ckpt = torch.load(vae_path, map_location=args.device, weights_only=False)
        vae.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded VAE from {vae_path}")
    else:
        print(f"  WARNING: {vae_path} not found, using untrained VAE")

    print("\n[2] Loading MLP Refiner...")

    # Find most recent refiner (filter to directories only, exclude files)
    refiner_dirs = [d for d in Path("outputs").glob("refiners_*") if d.is_dir()]
    if refiner_dirs:
        refiner_dir = sorted(refiner_dirs)[-1] / "mlp_refiner"
    else:
        refiner_dir = Path("outputs/refiners_20260129_230857/mlp_refiner")

    refiner_path = refiner_dir / "best.pt"
    print(f"  Looking for refiner at: {refiner_path}")
    if refiner_path.exists():
        ckpt = torch.load(refiner_path, map_location=args.device, weights_only=False)
        refiner_config = ckpt.get("config", RefinerConfig(latent_dim=32, hidden_dims=[64, 64, 32]))
        refiner = DDGMLPRefiner(config=refiner_config)
        refiner.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded MLP Refiner from {refiner_path}")
    else:
        print(f"  WARNING: {refiner_path} not found, using untrained refiner")
        refiner = DDGMLPRefiner(config=RefinerConfig(latent_dim=32, hidden_dims=[64, 64, 32]))

    # =========================================================
    # Load ProTherm dataset and extract embeddings
    # =========================================================
    print("\n[3] Extracting VAE+Refiner embeddings from ProTherm...")

    protherm_loader = ProThermLoader()
    protherm_db = protherm_loader.load_curated()
    base_dataset = protherm_loader.create_dataset(protherm_db)

    embedding_dataset = RefinerEmbeddingDataset(vae, refiner, base_dataset, args.device)
    print(f"  Extracted {len(embedding_dataset)} embeddings, dim={embedding_dataset.embedding_dim}")

    # Check baseline performance of MLP Refiner
    refiner_preds = embedding_dataset.refiner_preds.numpy()
    labels = embedding_dataset.labels.numpy()
    baseline_spearman = spearmanr(refiner_preds, labels)[0]
    print(f"  MLP Refiner baseline: Spearman {baseline_spearman:.4f}")

    # Split
    n_val = int(len(embedding_dataset) * 0.2)
    n_train = len(embedding_dataset) - n_val

    train_ds, val_ds = random_split(embedding_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    # Wrap to preserve embedding_dim
    train_ds = SubsetWrapper(train_ds, embedding_dataset.embedding_dim)
    val_ds = SubsetWrapper(val_ds, embedding_dataset.embedding_dim)

    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    # =========================================================
    # Train Stochastic Transformer
    # =========================================================
    print("\n[4] Training Stochastic Transformer...")

    config = StochasticTransformerConfig(
        d_model=64,
        n_heads=4,
        n_layers=4,
        d_ff=256,
        dropout=0.1,
        use_residual_from_refiner=True,
        use_multi_head_output=True,
    )

    model, history = train_stochastic_transformer(
        train_ds,
        val_ds,
        output_dir=output_dir,
        config=config,
        epochs=epochs,
        batch_size=16,
        lr=5e-5,
        patience=patience,
        device=args.device,
    )

    # =========================================================
    # QA: Standard evaluation
    # =========================================================
    print("\n[5] Quality Assurance...")

    qa_metrics = compute_qa_metrics(model, val_ds, args.device, use_mc=False)
    print_qa_report("Stochastic Transformer (deterministic)", qa_metrics)

    # QA: Monte Carlo evaluation
    mc_metrics = compute_qa_metrics(model, val_ds, args.device, use_mc=True)
    print_qa_report("Stochastic Transformer (MC dropout)", mc_metrics)

    # Save QA results
    qa_results = {
        "deterministic": qa_metrics,
        "monte_carlo": mc_metrics,
        "baseline_refiner_spearman": float(baseline_spearman),
    }

    with open(output_dir / "qa_results.json", "w") as f:
        json.dump(qa_results, f, indent=2)

    # =========================================================
    # Summary
    # =========================================================
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"  MLP Refiner baseline:    {baseline_spearman:.4f}")
    print(f"  Stochastic Transformer:  {qa_metrics['spearman']:.4f}")
    print(f"  Improvement:             {(qa_metrics['spearman'] - baseline_spearman) * 100:.1f}%")

    if qa_metrics["spearman"] > baseline_spearman:
        print("\n  ✓ Transformer IMPROVED over MLP Refiner!")
    else:
        print("\n  × Transformer did not improve over MLP Refiner")

    print(f"\n  Results saved to: {output_dir}")
    print("\n  Next: Run train_combined_transformer.py for combined filtered dataset")


if __name__ == "__main__":
    main()
