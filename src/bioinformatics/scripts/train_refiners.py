#!/usr/bin/env python3
"""Train MLP Refiner and Transformers using VAE embeddings.

The VAE embeddings provide a "topological shortcut" - a continuous/fuzzy
representation that helps discrete systems (MLP, Transformers) navigate
the mutation landscape and discover non-evident paths between mutations.

Usage:
    python src/bioinformatics/scripts/train_refiners.py [--quick]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))

import argparse
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split

from src.bioinformatics.data.protherm_loader import ProThermLoader
from src.bioinformatics.models.ddg_mlp_refiner import DDGMLPRefiner, RefinerConfig
from src.bioinformatics.models.ddg_transformer import TransformerConfig
from src.bioinformatics.models.ddg_vae import DDGVAE
from src.bioinformatics.training.deterministic import set_deterministic_mode


class VAEEmbeddingDataset(Dataset):
    """Dataset of VAE embeddings for downstream training."""

    def __init__(
        self,
        vae: DDGVAE,
        base_dataset: Dataset,
        device: str = "cuda",
    ):
        """Extract VAE embeddings from base dataset.

        Args:
            vae: Trained VAE model
            base_dataset: Dataset with (features, labels)
            device: Device for VAE inference
        """
        self.labels = []
        self.embeddings = []
        self.vae_preds = []

        vae = vae.to(device)
        vae.eval()

        # Extract embeddings
        with torch.no_grad():
            for i in range(len(base_dataset)):
                x, y = base_dataset[i]
                x = x.unsqueeze(0).to(device)

                output = vae(x)

                # Use mu (mean of latent distribution) as embedding
                # Falls back to z_hyp for hyperbolic models
                if "mu" in output:
                    self.embeddings.append(output["mu"].cpu().squeeze(0))
                else:
                    self.embeddings.append(output["z_hyp"].cpu().squeeze(0))
                self.vae_preds.append(output["ddg_pred"].cpu().squeeze())
                self.labels.append(y)

        self.embeddings = torch.stack(self.embeddings)
        self.vae_preds = torch.stack(self.vae_preds)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "embedding": self.embeddings[idx],
            "vae_pred": self.vae_preds[idx],
            "label": self.labels[idx],
        }

    @property
    def embedding_dim(self):
        return self.embeddings.shape[1]


def train_mlp_refiner(
    train_dataset: VAEEmbeddingDataset,
    val_dataset: VAEEmbeddingDataset,
    output_dir: Path,
    config: RefinerConfig | None = None,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = "cuda",
    verbose: bool = True,
) -> DDGMLPRefiner:
    """Train MLP Refiner on VAE embeddings.

    The refiner learns to correct/refine VAE predictions using the
    latent embeddings as a topological guide.
    """
    if config is None:
        config = RefinerConfig(
            latent_dim=train_dataset.embedding_dim,
            hidden_dims=[128, 128, 64, 32],
            dropout=0.1,
            use_residual=True,
        )

    model = DDGMLPRefiner(config=config).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    output_dir.mkdir(parents=True, exist_ok=True)

    best_spearman = -float("inf")
    history = {"train_loss": [], "val_loss": [], "val_spearman": []}

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            z = batch["embedding"].to(device)
            vae_pred = batch["vae_pred"].to(device).unsqueeze(-1)
            y = batch["label"].to(device).unsqueeze(-1)

            optimizer.zero_grad()
            loss_dict = model.loss(z, y, vae_pred)
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
                z = batch["embedding"].to(device)
                vae_pred = batch["vae_pred"].to(device).unsqueeze(-1)
                y = batch["label"].to(device).unsqueeze(-1)

                loss_dict = model.loss(z, y, vae_pred)
                val_loss += loss_dict["loss"].item()

                output = model(z, vae_pred)
                all_preds.extend(output["ddg_pred"].cpu().numpy().flatten())
                all_labels.extend(y.cpu().numpy().flatten())

        val_loss /= len(val_loader)
        val_spearman = spearmanr(all_preds, all_labels)[0]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_spearman"].append(val_spearman)

        if verbose and epoch % 10 == 0:
            rw = torch.sigmoid(model.residual_weight).item() if hasattr(model, "residual_weight") else 0
            print(
                f"Epoch {epoch:3d}: loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"spearman={val_spearman:.4f} res_weight={rw:.3f}"
            )

        if val_spearman > best_spearman:
            best_spearman = val_spearman
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "spearman": val_spearman,
                },
                output_dir / "best.pt",
            )

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

    return model


def train_transformer(
    train_dataset: VAEEmbeddingDataset,
    val_dataset: VAEEmbeddingDataset,
    output_dir: Path,
    model_type: str = "hierarchical",  # "full" or "hierarchical"
    config: TransformerConfig | None = None,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-4,
    device: str = "cuda",
    verbose: bool = True,
) -> nn.Module:
    """Train Transformer on VAE embeddings.

    Since we don't have full sequences, we treat the VAE embeddings
    as a pseudo-sequence representation. This allows the transformer
    to learn attention patterns over the embedding dimensions.
    """
    if config is None:
        config = TransformerConfig(
            max_seq_len=train_dataset.embedding_dim,  # Treat embedding dims as sequence
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=256,
            use_gradient_checkpointing=True,
            vocab_size=1,  # Not used - we use raw embeddings
        )

    # Create a transformer that takes embeddings directly
    class EmbeddingTransformer(nn.Module):
        """Transformer that processes VAE embeddings as pseudo-sequences."""

        def __init__(self, config, embedding_dim):
            super().__init__()
            self.config = config

            # Project embedding to transformer dimension
            self.input_proj = nn.Linear(1, config.d_model)

            # Positional encoding for embedding dimensions
            self.pos_enc = nn.Parameter(torch.randn(1, embedding_dim, config.d_model) * 0.02)

            # Transformer blocks
            self.blocks = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=config.d_model,
                        nhead=config.n_heads,
                        dim_feedforward=config.d_ff,
                        dropout=config.dropout,
                        activation="gelu",
                        batch_first=True,
                    )
                    for _ in range(config.n_layers)
                ]
            )

            # Prediction head
            self.head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.SiLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model, 1),
            )

        def forward(self, z, vae_pred=None):
            # z: (batch, embedding_dim)
            # Treat each dimension as a token
            x = z.unsqueeze(-1)  # (batch, embedding_dim, 1)
            x = self.input_proj(x)  # (batch, embedding_dim, d_model)
            x = x + self.pos_enc

            for block in self.blocks:
                x = block(x)

            # Global pooling
            x = x.mean(dim=1)  # (batch, d_model)

            # Predict delta from VAE prediction
            delta = self.head(x)

            if vae_pred is not None:
                return {"ddg_pred": vae_pred + delta, "delta": delta}
            return {"ddg_pred": delta, "delta": delta}

    model = EmbeddingTransformer(config, train_dataset.embedding_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    output_dir.mkdir(parents=True, exist_ok=True)

    best_spearman = -float("inf")
    history = {"train_loss": [], "val_loss": [], "val_spearman": []}

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            z = batch["embedding"].to(device)
            vae_pred = batch["vae_pred"].to(device).unsqueeze(-1)
            y = batch["label"].to(device).unsqueeze(-1)

            optimizer.zero_grad()
            output = model(z, vae_pred)
            loss = F.mse_loss(output["ddg_pred"], y)
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
                z = batch["embedding"].to(device)
                vae_pred = batch["vae_pred"].to(device).unsqueeze(-1)
                y = batch["label"].to(device).unsqueeze(-1)

                output = model(z, vae_pred)
                val_loss += F.mse_loss(output["ddg_pred"], y).item()

                all_preds.extend(output["ddg_pred"].cpu().numpy().flatten())
                all_labels.extend(y.cpu().numpy().flatten())

        val_loss /= len(val_loader)
        val_spearman = spearmanr(all_preds, all_labels)[0]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_spearman"].append(val_spearman)

        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: loss={train_loss:.4f} val_loss={val_loss:.4f} spearman={val_spearman:.4f}")

        if val_spearman > best_spearman:
            best_spearman = val_spearman
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "spearman": val_spearman,
                },
                output_dir / "best.pt",
            )

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

    return model


def main():
    parser = argparse.ArgumentParser(description="Train MLP Refiner and Transformers")
    parser.add_argument("--quick", action="store_true", help="Quick test run")
    parser.add_argument("--device", default="cuda", help="Device")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    set_deterministic_mode(42)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output = Path(f"outputs/refiners_{timestamp}")

    print("=" * 70)
    print("MLP Refiner & Transformer Training")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Output: {base_output}")

    epochs = 20 if args.quick else 100

    # ========================================
    # Load trained VAEs and extract embeddings
    # ========================================
    print("\n[1] Loading trained VAEs...")

    # Load VAE-ProTherm (best performing)
    vae_protherm = DDGVAE.create_protherm_variant(use_hyperbolic=False)
    ckpt_path = Path("outputs/ddg_vae_training_20260129_212316/vae_protherm/best.pt")
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        vae_protherm.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded VAE-ProTherm from {ckpt_path}")
    else:
        print(f"  WARNING: {ckpt_path} not found, using untrained VAE")

    # Load ProTherm dataset
    print("\n[2] Creating VAE embedding datasets...")
    protherm_loader = ProThermLoader()
    protherm_db = protherm_loader.load_curated()
    protherm_base = protherm_loader.create_dataset(protherm_db)

    # Extract VAE embeddings
    embedding_dataset = VAEEmbeddingDataset(vae_protherm, protherm_base, args.device)
    print(f"  Extracted {len(embedding_dataset)} embeddings, dim={embedding_dataset.embedding_dim}")

    # Split
    n_val = int(len(embedding_dataset) * 0.2)
    n_train = len(embedding_dataset) - n_val
    train_emb, val_emb = random_split(embedding_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    # Wrap in proper datasets
    class SubsetWrapper(Dataset):
        def __init__(self, subset):
            self.subset = subset
            self.embedding_dim = subset.dataset.embedding_dim

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            return self.subset[idx]

    train_emb = SubsetWrapper(train_emb)
    val_emb = SubsetWrapper(val_emb)

    print(f"  Train: {len(train_emb)}, Val: {len(val_emb)}")

    # ========================================
    # Train MLP Refiner
    # ========================================
    print("\n" + "=" * 70)
    print("[3] Training MLP Refiner (learns delta corrections)")
    print("=" * 70)

    refiner_config = RefinerConfig(
        latent_dim=embedding_dataset.embedding_dim,
        hidden_dims=[64, 64, 32],
        dropout=0.1,
        use_residual=True,
        initial_residual_weight=0.3,
    )

    train_mlp_refiner(
        train_emb,
        val_emb,
        output_dir=base_output / "mlp_refiner",
        config=refiner_config,
        epochs=epochs,
        batch_size=16,
        lr=1e-3,
        device=args.device,
    )
    print("MLP Refiner training complete!")

    # ========================================
    # Train Embedding Transformer
    # ========================================
    print("\n" + "=" * 70)
    print("[4] Training Embedding Transformer (attention over embedding dims)")
    print("=" * 70)

    transformer_config = TransformerConfig(
        max_seq_len=embedding_dataset.embedding_dim,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.1,
    )

    train_transformer(
        train_emb,
        val_emb,
        output_dir=base_output / "embedding_transformer",
        config=transformer_config,
        epochs=epochs,
        batch_size=16,
        lr=1e-3,
        device=args.device,
    )
    print("Embedding Transformer training complete!")

    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    # Load and report results
    for name in ["mlp_refiner", "embedding_transformer"]:
        hist_path = base_output / name / "training_history.json"
        if hist_path.exists():
            with open(hist_path) as f:
                hist = json.load(f)
            best_spearman = max(hist["val_spearman"])
            print(f"\n{name}:")
            print(f"  Best Spearman: {best_spearman:.4f}")
            print(f"  Checkpoint: {base_output / name / 'best.pt'}")


if __name__ == "__main__":
    main()
