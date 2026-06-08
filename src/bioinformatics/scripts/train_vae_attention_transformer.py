#!/usr/bin/env python3
"""Transformer with Attention over VAE Activations.

This transformer performs cross-attention over activations from:
1. Three specialist VAEs (S669, ProTherm, Wide) + their MLP Refiners
2. Optionally: Best multimodal VAE

The transformer learns to attend to the most informative activations
from each specialist for each input, enabling adaptive fusion.

Architecture:
    Input Features
        ↓
    [Frozen] VAE-S669 → mu_s669, pred_s669 → [Frozen] Refiner-S669 → h_s669, refined_s669
    [Frozen] VAE-ProTherm → mu_protherm, pred_protherm → [Frozen] Refiner-ProTherm → h_protherm, refined_protherm
    [Frozen] VAE-Wide → mu_wide, pred_wide → [Frozen] Refiner-Wide → h_wide, refined_wide
        ↓
    Cross-Attention Transformer
        ↓
    Final DDG Prediction
"""

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
project_root = Path(__file__).parents[3]
sys.path.insert(0, str(project_root))

from src.bioinformatics.data.preprocessing import compute_features
from src.bioinformatics.data.protherm_loader import ProThermLoader
from src.bioinformatics.data.s669_loader import S669Loader


@dataclass
class AttentionConfig:
    """Configuration for attention transformer."""

    # VAE architecture (must match trained models)
    vae_hidden_dim: int = 128
    vae_latent_dim: int = 32
    refiner_hidden_dims: list[int] = None

    # Transformer
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 3
    dropout: float = 0.1

    # Training
    epochs: int = 200
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 30

    # Checkpoints (will be set dynamically)
    vae_suite_dir: str | None = None

    def __post_init__(self):
        if self.refiner_hidden_dims is None:
            self.refiner_hidden_dims = [64, 64, 32]


class DDGVAE(nn.Module):
    """VAE for DDG prediction (same as training script)."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h).clamp(-10, 2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(x)
        pred = self.decoder(mu).squeeze(-1)
        return {"mu": mu, "logvar": logvar, "pred": pred}


class MLPRefiner(nn.Module):
    """MLP that refines VAE predictions."""

    def __init__(self, latent_dim: int, hidden_dims: list[int] = None, dropout: float = 0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64, 32]

        layers = []
        in_dim = latent_dim
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, h_dim),
                    nn.SiLU(),
                    nn.LayerNorm(h_dim),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = h_dim

        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 1)
        self.residual_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, mu: torch.Tensor, vae_pred: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.mlp(mu)
        delta = self.head(h).squeeze(-1)
        w = torch.sigmoid(self.residual_weight)
        refined = vae_pred + w * delta
        return {"pred": refined, "delta": delta, "hidden": h, "weight": w}


class VAEAttentionTransformer(nn.Module):
    """Transformer that attends over VAE activations from multiple specialists."""

    def __init__(
        self,
        vaes: dict[str, DDGVAE],
        refiners: dict[str, MLPRefiner],
        config: AttentionConfig,
    ):
        super().__init__()

        # Store frozen specialists
        self.vaes = nn.ModuleDict(vaes)
        self.refiners = nn.ModuleDict(refiners)

        # Freeze all specialists
        for vae in self.vaes.values():
            for param in vae.parameters():
                param.requires_grad = False
        for refiner in self.refiners.values():
            for param in refiner.parameters():
                param.requires_grad = False

        # Calculate embedding dimensions per specialist
        # Each specialist contributes: mu (32) + refiner_hidden (32) + 2 predictions
        self.n_specialists = len(vaes)
        self.embed_per_specialist = config.vae_latent_dim + config.refiner_hidden_dims[-1] + 2

        # Project each specialist's features to d_model
        self.specialist_projections = nn.ModuleDict(
            {name: nn.Linear(self.embed_per_specialist, config.d_model) for name in vaes}
        )

        # Learnable specialist embeddings (like positional encoding for modalities)
        self.specialist_embeddings = nn.ParameterDict(
            {name: nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02) for name in vaes}
        )

        # CLS token for final prediction
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # Output head
        self.head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 1),
        )

        # Also compute weighted average of specialist predictions for comparison
        self.specialist_weights = nn.Parameter(torch.ones(self.n_specialists) / self.n_specialists)

    def get_specialist_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Get features from all specialists."""
        features = {}

        with torch.no_grad():
            for name in self.vaes:
                vae = self.vaes[name]
                refiner = self.refiners[name]

                vae_out = vae(x)
                mu = vae_out["mu"]
                vae_pred = vae_out["pred"]

                refiner_out = refiner(mu, vae_pred)
                h = refiner_out["hidden"]
                refined_pred = refiner_out["pred"]

                # Concatenate all features: mu, hidden, vae_pred, refined_pred
                features[name] = torch.cat(
                    [
                        mu,
                        h,
                        vae_pred.unsqueeze(-1),
                        refined_pred.unsqueeze(-1),
                    ],
                    dim=-1,
                )

        return features

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = x.size(0)

        # Get features from all specialists
        specialist_features = self.get_specialist_features(x)

        # Project and add specialist embeddings
        projected = []
        specialist_preds = []

        for name in self.vaes:
            feat = specialist_features[name]
            proj = self.specialist_projections[name](feat)
            emb = self.specialist_embeddings[name].expand(batch_size, -1, -1)
            projected.append(proj.unsqueeze(1) + emb)

            # Also collect specialist predictions for weighted average
            specialist_preds.append(feat[:, -1])  # refined_pred is last

        # Stack specialists: [batch, n_specialists, d_model]
        x_specialists = torch.cat(projected, dim=1)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x_specialists], dim=1)

        # Transformer
        x = self.transformer(x)

        # Use CLS output for prediction
        cls_out = x[:, 0]
        transformer_pred = self.head(cls_out).squeeze(-1)

        # Also compute weighted average
        weights = F.softmax(self.specialist_weights, dim=0)
        specialist_preds = torch.stack(specialist_preds, dim=1)
        weighted_avg = (specialist_preds * weights).sum(dim=1)

        return {
            "pred": transformer_pred,
            "weighted_avg": weighted_avg,
            "specialist_preds": specialist_preds,
            "weights": weights,
            "cls_embedding": cls_out,
        }


def load_data():
    """Load S669 and ProTherm datasets."""
    # S669
    s669_loader = S669Loader()
    s669_records = s669_loader.load_from_csv()

    X_s669, y_s669 = [], []
    for record in s669_records:
        feat = compute_features(record.wild_type, record.mutant)
        X_s669.append(feat.to_array(include_hyperbolic=False))
        y_s669.append(record.ddg)
    X_s669 = np.array(X_s669, dtype=np.float32)
    y_s669 = np.array(y_s669, dtype=np.float32)

    # ProTherm
    protherm_loader = ProThermLoader()
    protherm_db = protherm_loader.load_curated()

    X_protherm, y_protherm = [], []
    for record in protherm_db.records:
        feat = compute_features(record.wild_type, record.mutant)
        X_protherm.append(feat.to_array(include_hyperbolic=False))
        y_protherm.append(record.ddg)
    X_protherm = np.array(X_protherm, dtype=np.float32)
    y_protherm = np.array(y_protherm, dtype=np.float32)

    return X_s669, y_s669, X_protherm, y_protherm


def load_specialists(suite_dir: Path, config: AttentionConfig, input_dim: int, device: torch.device):
    """Load trained VAEs and refiners."""
    vaes = {}
    refiners = {}

    for name in ["s669", "protherm"]:
        model_dir = suite_dir / name

        if not model_dir.exists():
            print(f"  Warning: {name} models not found at {model_dir}")
            continue

        # Load VAE
        vae = DDGVAE(
            input_dim=input_dim,
            hidden_dim=config.vae_hidden_dim,
            latent_dim=config.vae_latent_dim,
        ).to(device)
        vae_ckpt = torch.load(model_dir / "vae.pt", map_location=device, weights_only=False)
        vae.load_state_dict(vae_ckpt["model_state_dict"])
        vaes[name] = vae

        # Load Refiner
        refiner = MLPRefiner(
            latent_dim=config.vae_latent_dim,
            hidden_dims=config.refiner_hidden_dims,
        ).to(device)
        refiner_ckpt = torch.load(model_dir / "refiner.pt", map_location=device, weights_only=False)
        refiner.load_state_dict(refiner_ckpt["model_state_dict"])
        refiners[name] = refiner

        print(
            f"  Loaded {name}: VAE Spearman={vae_ckpt.get('vae_spearman', 'N/A'):.4f}, "
            f"Refiner Spearman={refiner_ckpt.get('refiner_spearman', 'N/A'):.4f}"
        )

    # Check for Wide (optional)
    wide_dir = suite_dir / "wide"
    if wide_dir.exists():
        try:
            vae = DDGVAE(
                input_dim=input_dim,
                hidden_dim=config.vae_hidden_dim,
                latent_dim=config.vae_latent_dim,
            ).to(device)
            vae_ckpt = torch.load(wide_dir / "vae.pt", map_location=device, weights_only=False)
            vae.load_state_dict(vae_ckpt["model_state_dict"])
            vaes["wide"] = vae

            refiner = MLPRefiner(
                latent_dim=config.vae_latent_dim,
                hidden_dims=config.refiner_hidden_dims,
            ).to(device)
            refiner_ckpt = torch.load(wide_dir / "refiner.pt", map_location=device, weights_only=False)
            refiner.load_state_dict(refiner_ckpt["model_state_dict"])
            refiners["wide"] = refiner

            print(f"  Loaded wide: VAE Spearman={vae_ckpt.get('vae_spearman', 'N/A'):.4f}")
        except Exception as e:
            print(f"  Wide loading failed: {e}")

    return vaes, refiners


def train_attention_transformer(
    model: VAEAttentionTransformer,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: AttentionConfig,
) -> tuple[VAEAttentionTransformer, dict]:
    """Train the attention transformer."""
    device = next(model.parameters()).device

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Only train transformer parameters
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_val_corr = -1
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_corr": [], "weighted_avg_corr": []}

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0

        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_x)

            # MSE loss
            loss = F.mse_loss(out["pred"], batch_y)

            # Ranking loss for correlation
            if len(batch_y) > 1:
                idx1 = torch.randperm(len(batch_y))[: min(32, len(batch_y))]
                idx2 = torch.randperm(len(batch_y))[: min(32, len(batch_y))]
                diff_pred = out["pred"][idx1] - out["pred"][idx2]
                diff_true = batch_y[idx1] - batch_y[idx2]
                ranking_loss = F.relu(0.1 - diff_pred * torch.sign(diff_true)).mean()
                loss = loss + 0.2 * ranking_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t)
            val_preds = val_out["pred"].cpu().numpy()
            weighted_preds = val_out["weighted_avg"].cpu().numpy()
            val_corr = spearmanr(y_val_t.cpu().numpy(), val_preds)[0]
            weighted_corr = spearmanr(y_val_t.cpu().numpy(), weighted_preds)[0]

        history["train_loss"].append(epoch_loss / len(loader))
        history["val_corr"].append(val_corr)
        history["weighted_avg_corr"].append(weighted_corr)

        if val_corr > best_val_corr:
            best_val_corr = val_corr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"    Early stopping at epoch {epoch}")
                break

        if (epoch + 1) % 20 == 0:
            weights = val_out["weights"].cpu().numpy()
            print(f"    Epoch {epoch + 1}: val_corr={val_corr:.4f}, weighted={weighted_corr:.4f}, weights={weights}")

    if best_state:
        model.load_state_dict(best_state)

    return model, {"best_val_corr": best_val_corr, "history": history}


def evaluate(model: VAEAttentionTransformer, X: np.ndarray, y: np.ndarray) -> dict:
    """Evaluate model."""
    device = next(model.parameters()).device
    X_t = torch.tensor(X, dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        out = model(X_t)

    preds = out["pred"].cpu().numpy()
    weighted = out["weighted_avg"].cpu().numpy()
    out["specialist_preds"].cpu().numpy()
    weights = out["weights"].cpu().numpy()

    return {
        "spearman": spearmanr(y, preds)[0],
        "pearson": pearsonr(y, preds)[0],
        "mae": np.mean(np.abs(y - preds)),
        "weighted_avg_spearman": spearmanr(y, weighted)[0],
        "specialist_weights": weights.tolist(),
        "n": len(y),
    }


def main():
    """Train attention transformer over VAE activations."""
    print("=" * 70)
    print("VAE ATTENTION TRANSFORMER")
    print("Attending over activations from specialist VAEs + Refiners")
    print("=" * 70)

    # Find most recent VAE suite
    suite_dirs = sorted(Path("outputs").glob("full_vae_suite_*"))
    if not suite_dirs:
        print("ERROR: No trained VAE suite found. Run train_full_vae_suite.py first.")
        return

    suite_dir = suite_dirs[-1]
    print(f"Using VAE suite: {suite_dir}")

    config = AttentionConfig(vae_suite_dir=str(suite_dir))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs/vae_attention_transformer_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    np.random.seed(42)
    torch.manual_seed(42)

    # Load data
    print("\n[1] Loading data...")
    X_s669, y_s669, X_protherm, y_protherm = load_data()
    print(f"  S669: {len(X_s669)} samples")
    print(f"  ProTherm: {len(X_protherm)} samples")

    input_dim = X_s669.shape[1]

    # Load specialists
    print("\n[2] Loading trained specialists...")
    vaes, refiners = load_specialists(suite_dir, config, input_dim, device)

    if len(vaes) < 2:
        print("ERROR: Need at least 2 specialists (S669 + ProTherm)")
        return

    # Create attention transformer
    print("\n[3] Creating attention transformer...")
    model = VAEAttentionTransformer(vaes, refiners, config).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Trainable: {trainable:,}, Frozen: {frozen:,}")

    # Prepare combined data
    X_combined = np.vstack([X_s669, X_protherm])
    y_combined = np.concatenate([y_s669, y_protherm])

    idx = np.random.permutation(len(X_combined))
    split = int(0.8 * len(idx))
    X_train, y_train = X_combined[idx[:split]], y_combined[idx[:split]]
    X_val, y_val = X_combined[idx[split:]], y_combined[idx[split:]]

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}")

    # Train
    print("\n[4] Training attention transformer...")
    model, train_history = train_attention_transformer(model, X_train, y_train, X_val, y_val, config)

    # Evaluate
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    # On combined validation
    combined_result = evaluate(model, X_val, y_val)
    print(f"\nCombined Validation (n={combined_result['n']}):")
    print(f"  Transformer Spearman: {combined_result['spearman']:.4f}")
    print(f"  Weighted Avg Spearman: {combined_result['weighted_avg_spearman']:.4f}")
    print(f"  Specialist weights: {combined_result['specialist_weights']}")

    # On each dataset separately
    s669_idx = np.random.permutation(len(X_s669))
    s669_val_idx = s669_idx[int(0.8 * len(s669_idx)) :]
    s669_result = evaluate(model, X_s669[s669_val_idx], y_s669[s669_val_idx])

    protherm_idx = np.random.permutation(len(X_protherm))
    protherm_val_idx = protherm_idx[int(0.8 * len(protherm_idx)) :]
    protherm_result = evaluate(model, X_protherm[protherm_val_idx], y_protherm[protherm_val_idx])

    print(f"\nS669 Validation (n={s669_result['n']}):")
    print(f"  Spearman: {s669_result['spearman']:.4f}")

    print(f"\nProTherm Validation (n={protherm_result['n']}):")
    print(f"  Spearman: {protherm_result['spearman']:.4f}")

    # Save results
    results = {
        "combined": combined_result,
        "s669": s669_result,
        "protherm": protherm_result,
        "config": config.__dict__,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    torch.save({"model_state_dict": model.state_dict(), **results}, output_dir / "model.pt")

    print(f"\nResults saved to: {output_dir}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
VAE Attention Transformer Results:
  Combined: {combined_result["spearman"]:.4f} Spearman
  S669:     {s669_result["spearman"]:.4f} Spearman
  ProTherm: {protherm_result["spearman"]:.4f} Spearman

Specialist Weights: {combined_result["specialist_weights"]}
""")

    return model, results


if __name__ == "__main__":
    main()
