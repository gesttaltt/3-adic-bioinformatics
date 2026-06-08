#!/usr/bin/env python3
"""Train Full Pipeline: VAE → Transformer → MLP Refiner.

Architecture:
    Raw Features → VAE (latent embeddings) → Transformer → MLP Refiner → DDG

The VAE provides "topological shortcuts" - continuous latent representations
where similar mutations cluster together, enabling the transformer to discover
non-evident paths between mutations.

Pipeline:
1. Train VAE on full combined data (S669 + ProTherm)
2. Extract VAE embeddings (mu) as latent features
3. Train Transformer on VAE embeddings
4. Train MLP Refiner on Transformer outputs
5. End-to-end fine-tuning (optional)

Usage:
    python src/bioinformatics/scripts/train_vae_transformer_refiner.py
    python src/bioinformatics/scripts/train_vae_transformer_refiner.py --quick
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split

from src.bioinformatics.data.preprocessing import compute_features
from src.bioinformatics.data.protherm_loader import ProThermLoader
from src.bioinformatics.data.s669_loader import S669Loader
from src.bioinformatics.models.ddg_vae import DDGVAE, DDGVAEConfig
from src.bioinformatics.training.deterministic import set_deterministic_mode


@dataclass
class PipelineConfig:
    """Configuration for the full VAE→Transformer→Refiner pipeline."""

    # VAE config
    vae_latent_dim: int = 32
    vae_hidden_dim: int = 128
    vae_n_layers: int = 2
    vae_dropout: float = 0.1
    vae_beta: float = 0.5  # KL weight

    # Transformer config
    trans_d_model: int = 64
    trans_n_heads: int = 4
    trans_n_layers: int = 4
    trans_dropout: float = 0.1

    # Refiner config
    refiner_hidden_dims: list[int] = field(default_factory=lambda: [64, 32])
    refiner_dropout: float = 0.1

    # Training
    vae_epochs: int = 100
    trans_epochs: int = 100
    refiner_epochs: int = 100
    finetune_epochs: int = 50
    batch_size: int = 32
    vae_lr: float = 1e-4
    trans_lr: float = 1e-4
    refiner_lr: float = 1e-3
    finetune_lr: float = 1e-5


class FullPipelineModel(nn.Module):
    """Complete VAE → Transformer → MLP Refiner pipeline."""

    def __init__(self, config: PipelineConfig, input_dim: int):
        super().__init__()
        self.config = config
        self.input_dim = input_dim

        # VAE for latent embeddings
        vae_config = DDGVAEConfig(
            input_dim=input_dim,
            hidden_dim=config.vae_hidden_dim,
            latent_dim=config.vae_latent_dim,
            n_layers=config.vae_n_layers,
            dropout=config.vae_dropout,
            beta=config.vae_beta,
            use_hyperbolic=False,
        )
        self.vae = DDGVAE(vae_config)

        # Transformer on VAE embeddings
        self.trans_input_proj = nn.Linear(1, config.trans_d_model)
        self.trans_pos_enc = nn.Parameter(torch.randn(1, config.vae_latent_dim, config.trans_d_model) * 0.02)
        self.trans_cls_token = nn.Parameter(torch.randn(1, 1, config.trans_d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.trans_d_model,
            nhead=config.trans_n_heads,
            dim_feedforward=config.trans_d_model * 4,
            dropout=config.trans_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.trans_n_layers,
        )

        # Transformer output head
        self.trans_head = nn.Sequential(
            nn.Linear(config.trans_d_model, config.trans_d_model),
            nn.SiLU(),
            nn.Dropout(config.trans_dropout),
            nn.Linear(config.trans_d_model, 1),
        )

        # MLP Refiner
        refiner_layers = []
        in_dim = config.trans_d_model + config.vae_latent_dim + 1  # CLS + mu + trans_pred
        for hidden_dim in config.refiner_hidden_dims:
            refiner_layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.SiLU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(config.refiner_dropout),
                ]
            )
            in_dim = hidden_dim
        refiner_layers.append(nn.Linear(in_dim, 1))
        self.refiner = nn.Sequential(*refiner_layers)

        # Learned residual weights for combining predictions
        self.vae_weight = nn.Parameter(torch.tensor(0.3))
        self.trans_weight = nn.Parameter(torch.tensor(0.3))
        self.refiner_weight = nn.Parameter(torch.tensor(0.4))

    def forward_vae(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass through VAE only."""
        return self.vae(x)

    def forward_transformer(self, mu: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass through transformer on VAE embeddings."""
        batch_size = mu.shape[0]

        # Treat each latent dimension as a token
        x = mu.unsqueeze(-1)  # (batch, latent_dim, 1)
        x = self.trans_input_proj(x)  # (batch, latent_dim, d_model)
        x = x + self.trans_pos_enc

        # Add CLS token
        cls_tokens = self.trans_cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Transform
        x = self.transformer(x)

        # Get CLS output
        cls_out = x[:, 0]

        # Predict
        trans_pred = self.trans_head(cls_out)

        return {
            "cls_embedding": cls_out,
            "trans_pred": trans_pred,
        }

    def forward_refiner(
        self,
        cls_embedding: torch.Tensor,
        mu: torch.Tensor,
        trans_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through MLP refiner."""
        # Concatenate all features
        combined = torch.cat([cls_embedding, mu, trans_pred], dim=-1)
        return self.refiner(combined)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Full forward pass through entire pipeline."""
        # VAE
        vae_out = self.forward_vae(x)
        mu = vae_out["mu"]
        vae_pred = vae_out["ddg_pred"]

        # Transformer
        trans_out = self.forward_transformer(mu)
        cls_embedding = trans_out["cls_embedding"]
        trans_pred = trans_out["trans_pred"]

        # Refiner
        refiner_pred = self.forward_refiner(cls_embedding, mu, trans_pred)

        # Weighted combination
        weights = F.softmax(torch.stack([self.vae_weight, self.trans_weight, self.refiner_weight]), dim=0)

        final_pred = weights[0] * vae_pred + weights[1] * trans_pred + weights[2] * refiner_pred

        return {
            "ddg_pred": final_pred,
            "vae_pred": vae_pred,
            "trans_pred": trans_pred,
            "refiner_pred": refiner_pred,
            "mu": mu,
            "logvar": vae_out["logvar"],
            "cls_embedding": cls_embedding,
            "weights": weights,
        }

    def loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        stage: str = "full",  # "vae", "transformer", "refiner", "full"
    ) -> dict[str, torch.Tensor]:
        """Compute loss for specified training stage."""
        if y.dim() == 1:
            y = y.unsqueeze(-1)

        if stage == "vae":
            vae_out = self.forward_vae(x)
            recon_loss = F.mse_loss(vae_out["ddg_pred"], y)
            kl_loss = -0.5 * torch.mean(1 + vae_out["logvar"] - vae_out["mu"].pow(2) - vae_out["logvar"].exp())
            total_loss = recon_loss + self.config.vae_beta * kl_loss
            return {"loss": total_loss, "recon_loss": recon_loss, "kl_loss": kl_loss}

        elif stage == "transformer":
            with torch.no_grad():
                vae_out = self.forward_vae(x)
                mu = vae_out["mu"]
            trans_out = self.forward_transformer(mu)
            loss = F.mse_loss(trans_out["trans_pred"], y)
            return {"loss": loss}

        elif stage == "refiner":
            with torch.no_grad():
                vae_out = self.forward_vae(x)
                mu = vae_out["mu"]
                trans_out = self.forward_transformer(mu)
            refiner_pred = self.forward_refiner(trans_out["cls_embedding"], mu, trans_out["trans_pred"])
            loss = F.mse_loss(refiner_pred, y)
            return {"loss": loss}

        else:  # full
            out = self.forward(x)
            final_loss = F.mse_loss(out["ddg_pred"], y)
            vae_loss = F.mse_loss(out["vae_pred"], y)
            trans_loss = F.mse_loss(out["trans_pred"], y)
            refiner_loss = F.mse_loss(out["refiner_pred"], y)

            # KL for VAE regularization
            kl_loss = -0.5 * torch.mean(1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp())

            total_loss = final_loss + 0.1 * kl_loss

            return {
                "loss": total_loss,
                "final_loss": final_loss,
                "vae_loss": vae_loss,
                "trans_loss": trans_loss,
                "refiner_loss": refiner_loss,
                "kl_loss": kl_loss,
            }


class CombinedDataset(Dataset):
    """Combined dataset from S669 and ProTherm."""

    def __init__(self, s669_records, protherm_records):
        self.features = []
        self.labels = []
        self.sources = []

        # Process S669
        for record in s669_records:
            feat = compute_features(record.wild_type, record.mutant)
            self.features.append(torch.tensor(feat.to_array(include_hyperbolic=False), dtype=torch.float32))
            self.labels.append(record.ddg)
            self.sources.append(0)

        # Process ProTherm
        for record in protherm_records:
            if hasattr(record, "wild_type"):
                wt, mut, ddg = record.wild_type, record.mutant, record.ddg
            else:
                wt, mut, ddg = record["wild_type"], record["mutant"], record["ddg"]
            feat = compute_features(wt, mut)
            self.features.append(torch.tensor(feat.to_array(include_hyperbolic=False), dtype=torch.float32))
            self.labels.append(ddg)
            self.sources.append(1)

        self.features = torch.stack(self.features)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        self.sources = torch.tensor(self.sources, dtype=torch.long)

        # Standardize labels per source
        for src in [0, 1]:
            mask = self.sources == src
            if mask.sum() > 0:
                src_labels = self.labels[mask]
                mean, std = src_labels.mean(), src_labels.std()
                if std > 0:
                    self.labels[mask] = (src_labels - mean) / std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.sources[idx]

    @property
    def input_dim(self):
        return self.features.shape[1]


def train_stage(
    model: FullPipelineModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    stage: str,
    epochs: int,
    lr: float,
    patience: int,
    device: str,
    verbose: bool = True,
) -> dict[str, list[float]]:
    """Train a specific stage of the pipeline."""

    # Freeze/unfreeze appropriate parameters
    if stage == "vae":
        for p in model.vae.parameters():
            p.requires_grad = True
        for p in model.transformer.parameters():
            p.requires_grad = False
        for p in model.refiner.parameters():
            p.requires_grad = False
        params = model.vae.parameters()
    elif stage == "transformer":
        for p in model.vae.parameters():
            p.requires_grad = False
        for p in model.transformer.parameters():
            p.requires_grad = True
        for p in model.trans_input_proj.parameters():
            p.requires_grad = True
        for p in model.trans_head.parameters():
            p.requires_grad = True
        for p in model.refiner.parameters():
            p.requires_grad = False
        params = (
            list(model.transformer.parameters())
            + list(model.trans_input_proj.parameters())
            + list(model.trans_head.parameters())
            + [model.trans_pos_enc, model.trans_cls_token]
        )
    elif stage == "refiner":
        for p in model.vae.parameters():
            p.requires_grad = False
        for p in model.transformer.parameters():
            p.requires_grad = False
        for p in model.refiner.parameters():
            p.requires_grad = True
        params = model.refiner.parameters()
    else:  # full
        for p in model.parameters():
            p.requires_grad = True
        params = model.parameters()

    optimizer = AdamW(params, lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "val_loss": [], "val_spearman": []}
    best_spearman = -float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x, y, _ = batch
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            loss_dict = model.loss(x, y, stage=stage)
            loss = loss_dict["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                x, y, _ = batch
                x, y = x.to(device), y.to(device)

                if stage == "vae":
                    out = model.forward_vae(x)
                    pred = out["ddg_pred"]
                elif stage == "transformer":
                    vae_out = model.forward_vae(x)
                    trans_out = model.forward_transformer(vae_out["mu"])
                    pred = trans_out["trans_pred"]
                elif stage == "refiner":
                    vae_out = model.forward_vae(x)
                    trans_out = model.forward_transformer(vae_out["mu"])
                    pred = model.forward_refiner(trans_out["cls_embedding"], vae_out["mu"], trans_out["trans_pred"])
                else:
                    out = model.forward(x)
                    pred = out["ddg_pred"]

                loss = F.mse_loss(pred.squeeze(), y)
                val_loss += loss.item()

                all_preds.extend(pred.squeeze().cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        val_loss /= len(val_loader)
        val_spearman = spearmanr(all_preds, all_labels)[0]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_spearman"].append(float(val_spearman))

        if verbose and epoch % 10 == 0:
            print(f"  [{stage}] Epoch {epoch:3d}: loss={train_loss:.4f} val_loss={val_loss:.4f} ρ={val_spearman:.4f}")

        if val_spearman > best_spearman:
            best_spearman = val_spearman
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  [{stage}] Early stopping at epoch {epoch}")
                break

    # Restore best state
    if best_state is not None:
        model.load_state_dict(best_state)

    return history, best_spearman


def compute_qa_metrics(
    model: FullPipelineModel,
    dataloader: DataLoader,
    device: str,
) -> dict[str, float]:
    """Compute QA metrics."""
    model.eval()

    all_preds = []
    all_labels = []
    all_sources = []
    component_preds = {"vae": [], "trans": [], "refiner": [], "final": []}

    with torch.no_grad():
        for batch in dataloader:
            x, y, src = batch
            x = x.to(device)

            out = model.forward(x)

            all_preds.extend(out["ddg_pred"].squeeze().cpu().numpy())
            all_labels.extend(y.numpy())
            all_sources.extend(src.numpy())

            component_preds["vae"].extend(out["vae_pred"].squeeze().cpu().numpy())
            component_preds["trans"].extend(out["trans_pred"].squeeze().cpu().numpy())
            component_preds["refiner"].extend(out["refiner_pred"].squeeze().cpu().numpy())
            component_preds["final"].extend(out["ddg_pred"].squeeze().cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_sources = np.array(all_sources)

    # Overall metrics
    spearman_r, spearman_p = spearmanr(all_preds, all_labels)
    pearson_r, _ = pearsonr(all_preds, all_labels)
    mae = np.mean(np.abs(all_preds - all_labels))

    result = {
        "spearman": float(spearman_r),
        "spearman_p": float(spearman_p),
        "pearson": float(pearson_r),
        "mae": float(mae),
        "n_samples": len(all_preds),
    }

    # Per-source metrics
    for src_id, src_name in [(0, "s669"), (1, "protherm")]:
        mask = all_sources == src_id
        if mask.sum() > 2:
            src_preds = all_preds[mask]
            src_labels = all_labels[mask]
            src_r, _ = spearmanr(src_preds, src_labels)
            result[f"{src_name}_spearman"] = float(src_r)
            result[f"{src_name}_n"] = int(mask.sum())

    # Per-component metrics
    for comp_name, preds in component_preds.items():
        preds = np.array(preds)
        comp_r, _ = spearmanr(preds, all_labels)
        result[f"{comp_name}_spearman"] = float(comp_r)

    return result


def print_qa_report(name: str, metrics: dict[str, float]):
    """Print QA report."""
    print(f"\n{'=' * 60}")
    print(f"QA REPORT: {name}")
    print(f"{'=' * 60}")
    print(f"  Samples: {metrics['n_samples']}")
    print(f"  Overall Spearman: {metrics['spearman']:.4f}")
    print(f"  Overall Pearson:  {metrics['pearson']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")

    print("\n  Per-Source:")
    for src in ["s669", "protherm"]:
        if f"{src}_spearman" in metrics:
            print(f"    {src:12s}: ρ={metrics[f'{src}_spearman']:.4f} (n={metrics[f'{src}_n']})")

    print("\n  Per-Component:")
    for comp in ["vae", "trans", "refiner", "final"]:
        if f"{comp}_spearman" in metrics:
            print(f"    {comp:12s}: ρ={metrics[f'{comp}_spearman']:.4f}")

    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Train VAE→Transformer→Refiner Pipeline")
    parser.add_argument("--quick", action="store_true", help="Quick test run")
    parser.add_argument("--device", default="cuda", help="Device")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    set_deterministic_mode(42)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs/vae_transformer_refiner_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Config
    config = PipelineConfig()
    if args.quick:
        config.vae_epochs = 30
        config.trans_epochs = 30
        config.refiner_epochs = 30
        config.finetune_epochs = 20

    patience = 10 if args.quick else 25

    print("=" * 70)
    print("VAE → TRANSFORMER → MLP REFINER PIPELINE")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")

    # =========================================================
    # Load Data
    # =========================================================
    print("\n[1] Loading datasets...")

    s669_loader = S669Loader()
    s669_records = s669_loader.load_from_csv()
    print(f"  S669: {len(s669_records)} records")

    protherm_loader = ProThermLoader()
    protherm_records = protherm_loader.load_curated()
    print(f"  ProTherm: {len(protherm_records)} records")

    # Create combined dataset
    dataset = CombinedDataset(s669_records, protherm_records)
    print(f"  Combined: {len(dataset)} records, input_dim={dataset.input_dim}")

    # Split
    n_val = int(len(dataset) * 0.15)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    # =========================================================
    # Create Model
    # =========================================================
    print("\n[2] Creating model...")

    model = FullPipelineModel(config, dataset.input_dim).to(args.device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # =========================================================
    # Stage 1: Train VAE
    # =========================================================
    print("\n" + "=" * 60)
    print("[3] Stage 1: Training VAE (latent embeddings)")
    print("=" * 60)

    vae_history, vae_best = train_stage(
        model,
        train_loader,
        val_loader,
        stage="vae",
        epochs=config.vae_epochs,
        lr=config.vae_lr,
        patience=patience,
        device=args.device,
    )
    print(f"  VAE Best Spearman: {vae_best:.4f}")

    # =========================================================
    # Stage 2: Train Transformer
    # =========================================================
    print("\n" + "=" * 60)
    print("[4] Stage 2: Training Transformer (on VAE embeddings)")
    print("=" * 60)

    trans_history, trans_best = train_stage(
        model,
        train_loader,
        val_loader,
        stage="transformer",
        epochs=config.trans_epochs,
        lr=config.trans_lr,
        patience=patience,
        device=args.device,
    )
    print(f"  Transformer Best Spearman: {trans_best:.4f}")

    # =========================================================
    # Stage 3: Train MLP Refiner
    # =========================================================
    print("\n" + "=" * 60)
    print("[5] Stage 3: Training MLP Refiner")
    print("=" * 60)

    refiner_history, refiner_best = train_stage(
        model,
        train_loader,
        val_loader,
        stage="refiner",
        epochs=config.refiner_epochs,
        lr=config.refiner_lr,
        patience=patience,
        device=args.device,
    )
    print(f"  Refiner Best Spearman: {refiner_best:.4f}")

    # =========================================================
    # Stage 4: End-to-End Fine-tuning
    # =========================================================
    print("\n" + "=" * 60)
    print("[6] Stage 4: End-to-End Fine-tuning")
    print("=" * 60)

    finetune_history, finetune_best = train_stage(
        model,
        train_loader,
        val_loader,
        stage="full",
        epochs=config.finetune_epochs,
        lr=config.finetune_lr,
        patience=patience,
        device=args.device,
    )
    print(f"  Fine-tuned Best Spearman: {finetune_best:.4f}")

    # =========================================================
    # QA
    # =========================================================
    print("\n[7] Quality Assurance...")

    qa_metrics = compute_qa_metrics(model, val_loader, args.device)
    print_qa_report("Full Pipeline (VAE→Transformer→Refiner)", qa_metrics)

    # Get learned weights
    model.eval()
    with torch.no_grad():
        weights = F.softmax(torch.stack([model.vae_weight, model.trans_weight, model.refiner_weight]), dim=0)
    print("\n  Learned Weights:")
    print(f"    VAE:      {weights[0].item():.3f}")
    print(f"    Trans:    {weights[1].item():.3f}")
    print(f"    Refiner:  {weights[2].item():.3f}")

    # =========================================================
    # Save
    # =========================================================
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "qa_metrics": qa_metrics,
        },
        output_dir / "best.pt",
    )

    results = {
        "vae_best": vae_best,
        "trans_best": trans_best,
        "refiner_best": refiner_best,
        "finetune_best": finetune_best,
        "qa_metrics": qa_metrics,
        "weights": {
            "vae": float(weights[0]),
            "trans": float(weights[1]),
            "refiner": float(weights[2]),
        },
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # =========================================================
    # Summary
    # =========================================================
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 70)
    print("\n  Stage Results:")
    print(f"    VAE:        {vae_best:.4f}")
    print(f"    Transformer: {trans_best:.4f}")
    print(f"    Refiner:    {refiner_best:.4f}")
    print(f"    Fine-tuned: {finetune_best:.4f}")
    print(f"\n  Final Pipeline: {qa_metrics['spearman']:.4f}")
    print(f"    S669:     {qa_metrics.get('s669_spearman', 'N/A')}")
    print(f"    ProTherm: {qa_metrics.get('protherm_spearman', 'N/A')}")
    print(f"\n  Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
