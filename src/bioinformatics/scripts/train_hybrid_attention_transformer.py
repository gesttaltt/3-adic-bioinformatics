#!/usr/bin/env python3
"""Hybrid Attention Transformer over VAE + Transformer Specialists.

Addresses the weakness of S669 VAE (0.28) by incorporating Transformer-S669 (0.47)
alongside VAE specialists. The attention mechanism learns which specialist to trust
for different mutation types.

Architecture:
- S669: VAE (0.28) + Transformer (0.47) features
- ProTherm: VAE (0.85) + Refiner (0.89) features
- Wide: VAE + Refiner features (when available)
- Attention fuses all specialist features

Target: Beat individual models by selective attention.
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
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
project_root = Path(__file__).parents[3]
sys.path.insert(0, str(project_root))

from src.bioinformatics.data.preprocessing import compute_features
from src.bioinformatics.data.protherm_loader import ProThermLoader
from src.bioinformatics.data.s669_loader import S669Loader


@dataclass
class HybridConfig:
    """Configuration for hybrid attention transformer."""

    # VAE architecture (must match trained models)
    vae_hidden_dim: int = 128
    vae_latent_dim: int = 32
    vae_dropout: float = 0.1

    # Transformer architecture (must match trained models)
    transformer_d_model: int = 96
    transformer_n_layers: int = 4
    transformer_n_heads: int = 6

    # MLP Refiner architecture (must match trained models)
    refiner_hidden_dims: list[int] = None

    # Attention transformer
    attention_d_model: int = 128
    attention_n_layers: int = 4
    attention_n_heads: int = 8
    attention_dropout: float = 0.1

    # Training
    epochs: int = 200
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 30

    def __post_init__(self):
        if self.refiner_hidden_dims is None:
            self.refiner_hidden_dims = [64, 64, 32]


class DDGVAE(nn.Module):
    """VAE for DDG prediction (must match training architecture)."""

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
        self.final_hidden_dim = in_dim

    def forward(self, mu: torch.Tensor, vae_pred: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.mlp(mu)
        delta = self.head(h).squeeze(-1)
        w = torch.sigmoid(self.residual_weight)
        refined = vae_pred + w * delta
        return {"pred": refined, "hidden": h, "delta": delta, "weight": w}


class DDGTransformer(nn.Module):
    """Direct transformer for DDG prediction (from features)."""

    def __init__(self, input_dim: int, d_model: int = 96, n_layers: int = 4, n_heads: int = 6, dropout: float = 0.05):
        super().__init__()
        self.d_model = d_model

        # Feature embedding
        self.feature_embed = nn.Linear(1, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, input_dim + 1, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.SiLU(), nn.Dropout(dropout), nn.Linear(d_model, 1)
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        B = x.size(0)

        # Embed features
        x = x.unsqueeze(-1)  # (B, F, 1)
        x = self.feature_embed(x)  # (B, F, d_model)

        # Add CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, F+1, d_model)

        # Add positional embedding
        x = x + self.pos_embed[:, : x.size(1)]

        # Transformer
        h = self.encoder(x)
        cls_out = h[:, 0]  # CLS token output

        # Predict
        pred = self.head(cls_out).squeeze(-1)

        return {"pred": pred, "hidden": cls_out}


class HybridAttentionTransformer(nn.Module):
    """Attention transformer over hybrid VAE + Transformer specialists.

    Each specialist contributes:
    - VAE: mu (latent_dim) + pred (1)
    - Refiner: hidden (refiner_dim) + pred (1)
    - Transformer: hidden (d_model) + pred (1)

    For S669: VAE + Transformer features
    For ProTherm: VAE + Refiner features
    For Wide: VAE + Refiner features (when available)
    """

    def __init__(self, config: HybridConfig, specialist_dims: dict[str, int]):
        super().__init__()
        self.config = config
        self.specialist_dims = specialist_dims

        # Per-specialist projection to common dimension
        self.projections = nn.ModuleDict()
        for name, dim in specialist_dims.items():
            self.projections[name] = nn.Linear(dim, config.attention_d_model)

        # Learnable specialist embeddings (identify which specialist)
        self.specialist_embeddings = nn.ParameterDict(
            {name: nn.Parameter(torch.randn(1, 1, config.attention_d_model) * 0.02) for name in specialist_dims}
        )

        # CLS token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.attention_d_model) * 0.02)

        # Transformer for cross-specialist attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.attention_d_model,
            nhead=config.attention_n_heads,
            dim_feedforward=config.attention_d_model * 4,
            dropout=config.attention_dropout,
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        self.attention_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.attention_n_layers)

        # Output head
        self.head = nn.Sequential(
            nn.LayerNorm(config.attention_d_model),
            nn.Linear(config.attention_d_model, config.attention_d_model),
            nn.SiLU(),
            nn.Dropout(config.attention_dropout),
            nn.Linear(config.attention_d_model, 1),
        )

    def forward(self, specialist_features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass with features from all available specialists.

        Args:
            specialist_features: Dict mapping specialist name to features tensor
                Each tensor has shape (B, specialist_dim)

        Returns:
            Dict with 'pred' and attention weights
        """
        B = next(iter(specialist_features.values())).size(0)
        next(iter(specialist_features.values())).device

        # Project each specialist to common dimension
        projected = []
        specialist_names = []
        for name, features in specialist_features.items():
            if name in self.projections:
                proj = self.projections[name](features)  # (B, d_model)
                proj = proj.unsqueeze(1)  # (B, 1, d_model)
                proj = proj + self.specialist_embeddings[name]  # Add specialist ID
                projected.append(proj)
                specialist_names.append(name)

        # Concatenate all specialists
        tokens = torch.cat(projected, dim=1)  # (B, n_specialists, d_model)

        # Add CLS token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, n_specialists+1, d_model)

        # Cross-specialist attention
        h = self.attention_encoder(tokens)
        cls_out = h[:, 0]  # (B, d_model)

        # Final prediction
        pred = self.head(cls_out).squeeze(-1)

        return {"pred": pred, "hidden": cls_out, "specialists": specialist_names}


def load_checkpoints(output_dir: Path) -> dict:
    """Load all available specialist checkpoints."""
    checkpoints = {}

    # S669 VAE (from full training)
    s669_vae_path = output_dir / "s669" / "vae.pt"
    if s669_vae_path.exists():
        checkpoints["s669_vae"] = torch.load(s669_vae_path, map_location="cpu", weights_only=False)
        print(f"  Loaded S669 VAE: Spearman {checkpoints['s669_vae'].get('vae_spearman', 'N/A')}")

    # S669 Refiner
    s669_refiner_path = output_dir / "s669" / "refiner.pt"
    if s669_refiner_path.exists():
        checkpoints["s669_refiner"] = torch.load(s669_refiner_path, map_location="cpu", weights_only=False)
        print(f"  Loaded S669 Refiner: Spearman {checkpoints['s669_refiner'].get('refiner_spearman', 'N/A')}")

    # ProTherm VAE
    protherm_vae_path = output_dir / "protherm" / "vae.pt"
    if protherm_vae_path.exists():
        checkpoints["protherm_vae"] = torch.load(protherm_vae_path, map_location="cpu", weights_only=False)
        print(f"  Loaded ProTherm VAE: Spearman {checkpoints['protherm_vae'].get('vae_spearman', 'N/A')}")

    # ProTherm Refiner
    protherm_refiner_path = output_dir / "protherm" / "refiner.pt"
    if protherm_refiner_path.exists():
        checkpoints["protherm_refiner"] = torch.load(protherm_refiner_path, map_location="cpu", weights_only=False)
        print(f"  Loaded ProTherm Refiner: Spearman {checkpoints['protherm_refiner'].get('refiner_spearman', 'N/A')}")

    # Wide VAE (if available)
    wide_vae_path = output_dir / "wide" / "vae.pt"
    if wide_vae_path.exists():
        checkpoints["wide_vae"] = torch.load(wide_vae_path, map_location="cpu", weights_only=False)
        print(f"  Loaded Wide VAE: Spearman {checkpoints['wide_vae'].get('vae_spearman', 'N/A')}")

    # Wide Refiner
    wide_refiner_path = output_dir / "wide" / "refiner.pt"
    if wide_refiner_path.exists():
        checkpoints["wide_refiner"] = torch.load(wide_refiner_path, map_location="cpu", weights_only=False)
        print(f"  Loaded Wide Refiner: Spearman {checkpoints['wide_refiner'].get('refiner_spearman', 'N/A')}")

    # Check for trained transformers from earlier experiments
    transformer_dirs = [
        project_root / "outputs" / "transformers_parallel_20260130_135014",
        project_root / "outputs" / "transformers_parallel_20260130_140108",
    ]

    for tdir in transformer_dirs:
        s669_t_path = tdir / "transformer_s669" / "best.pt"
        if s669_t_path.exists() and "s669_transformer" not in checkpoints:
            try:
                checkpoints["s669_transformer"] = torch.load(s669_t_path, map_location="cpu", weights_only=False)
                print(f"  Loaded S669 Transformer from {tdir.name}")
            except Exception as e:
                print(f"  Could not load S669 Transformer: {e}")

        protherm_t_path = tdir / "transformer_protherm" / "best.pt"
        if protherm_t_path.exists() and "protherm_transformer" not in checkpoints:
            try:
                checkpoints["protherm_transformer"] = torch.load(
                    protherm_t_path, map_location="cpu", weights_only=False
                )
                print(f"  Loaded ProTherm Transformer from {tdir.name}")
            except Exception as e:
                print(f"  Could not load ProTherm Transformer: {e}")

    return checkpoints


def load_combined_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load combined S669 + ProTherm data."""
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
    source_s669 = np.zeros(len(y_s669), dtype=np.int64)

    # ProTherm
    protherm_loader = ProThermLoader()
    protherm_records = protherm_loader.load_curated()

    X_protherm, y_protherm = [], []
    for record in protherm_records:
        feat = compute_features(record.wild_type, record.mutant)
        X_protherm.append(feat.to_array(include_hyperbolic=False))
        y_protherm.append(record.ddg)

    X_protherm = np.array(X_protherm, dtype=np.float32)
    y_protherm = np.array(y_protherm, dtype=np.float32)
    source_protherm = np.ones(len(y_protherm), dtype=np.int64)

    # Combine
    X = np.concatenate([X_s669, X_protherm], axis=0)
    y = np.concatenate([y_s669, y_protherm], axis=0)
    source = np.concatenate([source_s669, source_protherm], axis=0)

    print(f"Combined data: {len(X)} samples (S669: {len(X_s669)}, ProTherm: {len(X_protherm)})")

    return X, y, source, (X_s669, y_s669, X_protherm, y_protherm)


def create_specialists(config: HybridConfig, input_dim: int) -> dict[str, nn.Module]:
    """Create specialist models."""
    specialists = {}

    # VAEs
    specialists["s669_vae"] = DDGVAE(
        input_dim=input_dim,
        hidden_dim=config.vae_hidden_dim,
        latent_dim=config.vae_latent_dim,
        dropout=config.vae_dropout,
    )

    specialists["protherm_vae"] = DDGVAE(
        input_dim=input_dim,
        hidden_dim=config.vae_hidden_dim,
        latent_dim=config.vae_latent_dim,
        dropout=config.vae_dropout,
    )

    # Refiners
    specialists["s669_refiner"] = MLPRefiner(
        latent_dim=config.vae_latent_dim, hidden_dims=config.refiner_hidden_dims, dropout=config.vae_dropout
    )

    specialists["protherm_refiner"] = MLPRefiner(
        latent_dim=config.vae_latent_dim, hidden_dims=config.refiner_hidden_dims, dropout=config.vae_dropout
    )

    # Direct transformers (for S669 to compensate weak VAE)
    specialists["s669_transformer"] = DDGTransformer(
        input_dim=input_dim,
        d_model=config.transformer_d_model,
        n_layers=config.transformer_n_layers,
        n_heads=config.transformer_n_heads,
    )

    return specialists


def load_specialist_weights(specialists: dict[str, nn.Module], checkpoints: dict) -> None:
    """Load trained weights into specialists."""
    for name, model in specialists.items():
        ckpt_key = name
        if ckpt_key in checkpoints:
            try:
                model.load_state_dict(checkpoints[ckpt_key]["model_state_dict"])
                print(f"  Loaded weights for {name}")
            except Exception as e:
                print(f"  Could not load weights for {name}: {e}")


def get_specialist_features(
    specialists: dict[str, nn.Module], x: torch.Tensor, config: HybridConfig
) -> dict[str, torch.Tensor]:
    """Get features from all specialists for attention."""
    features = {}

    with torch.no_grad():
        # S669 VAE + Refiner
        s669_vae_out = specialists["s669_vae"](x)
        s669_refiner_out = specialists["s669_refiner"](s669_vae_out["mu"], s669_vae_out["pred"])

        # S669 features: mu + refiner_hidden + predictions
        s669_feat = torch.cat(
            [
                s669_vae_out["mu"],
                s669_refiner_out["hidden"],
                s669_vae_out["pred"].unsqueeze(-1),
                s669_refiner_out["pred"].unsqueeze(-1),
            ],
            dim=-1,
        )
        features["s669_vae_refiner"] = s669_feat

        # S669 Transformer
        s669_t_out = specialists["s669_transformer"](x)
        s669_t_feat = torch.cat([s669_t_out["hidden"], s669_t_out["pred"].unsqueeze(-1)], dim=-1)
        features["s669_transformer"] = s669_t_feat

        # ProTherm VAE + Refiner
        protherm_vae_out = specialists["protherm_vae"](x)
        protherm_refiner_out = specialists["protherm_refiner"](protherm_vae_out["mu"], protherm_vae_out["pred"])

        protherm_feat = torch.cat(
            [
                protherm_vae_out["mu"],
                protherm_refiner_out["hidden"],
                protherm_vae_out["pred"].unsqueeze(-1),
                protherm_refiner_out["pred"].unsqueeze(-1),
            ],
            dim=-1,
        )
        features["protherm_vae_refiner"] = protherm_feat

    return features


def train_hybrid_attention(
    specialists: dict[str, nn.Module],
    attention_model: HybridAttentionTransformer,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: HybridConfig,
    device: torch.device,
) -> dict:
    """Train the hybrid attention transformer."""
    # Freeze specialists
    for model in specialists.values():
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    # Move to device
    for model in specialists.values():
        model.to(device)
    attention_model.to(device)

    # Prepare data
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=config.batch_size, shuffle=True)

    # Optimizer
    optimizer = torch.optim.AdamW(attention_model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    # Training loop
    best_spearman = -1
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_spearman": []}

    for epoch in range(config.epochs):
        attention_model.train()
        train_losses = []

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Get specialist features
            features = get_specialist_features(specialists, batch_x, config)

            # Attention forward
            out = attention_model(features)
            pred = out["pred"]

            loss = F.mse_loss(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(attention_model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        scheduler.step()

        # Validation
        attention_model.eval()
        with torch.no_grad():
            X_val_d = X_val_t.to(device)
            y_val_d = y_val_t.to(device)

            val_features = get_specialist_features(specialists, X_val_d, config)
            val_out = attention_model(val_features)
            val_pred = val_out["pred"]

            val_loss = F.mse_loss(val_pred, y_val_d).item()
            val_spearman = spearmanr(y_val, val_pred.cpu().numpy())[0]

        history["train_loss"].append(np.mean(train_losses))
        history["val_loss"].append(val_loss)
        history["val_spearman"].append(val_spearman)

        if val_spearman > best_spearman:
            best_spearman = val_spearman
            best_state = attention_model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or patience_counter == 0:
            print(
                f"  Epoch {epoch:3d}: loss={np.mean(train_losses):.4f}, "
                f"val_loss={val_loss:.4f}, val_spearman={val_spearman:.4f}"
                f"{' *' if patience_counter == 0 else ''}"
            )

        if patience_counter >= config.patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Restore best
    if best_state is not None:
        attention_model.load_state_dict(best_state)

    history["best_spearman"] = best_spearman
    return history


def evaluate_by_source(
    specialists: dict[str, nn.Module],
    attention_model: HybridAttentionTransformer,
    X_s669: np.ndarray,
    y_s669: np.ndarray,
    X_protherm: np.ndarray,
    y_protherm: np.ndarray,
    config: HybridConfig,
    device: torch.device,
) -> dict:
    """Evaluate on each dataset separately."""
    attention_model.eval()

    results = {}

    with torch.no_grad():
        # S669
        X_s669_t = torch.tensor(X_s669, dtype=torch.float32, device=device)
        features_s669 = get_specialist_features(specialists, X_s669_t, config)
        out_s669 = attention_model(features_s669)
        pred_s669 = out_s669["pred"].cpu().numpy()
        spearman_s669 = spearmanr(y_s669, pred_s669)[0]
        results["s669"] = {"spearman": spearman_s669, "n": len(y_s669)}

        # ProTherm
        X_protherm_t = torch.tensor(X_protherm, dtype=torch.float32, device=device)
        features_protherm = get_specialist_features(specialists, X_protherm_t, config)
        out_protherm = attention_model(features_protherm)
        pred_protherm = out_protherm["pred"].cpu().numpy()
        spearman_protherm = spearmanr(y_protherm, pred_protherm)[0]
        results["protherm"] = {"spearman": spearman_protherm, "n": len(y_protherm)}

        # Combined
        np.concatenate([X_s669, X_protherm])
        y_all = np.concatenate([y_s669, y_protherm])
        pred_all = np.concatenate([pred_s669, pred_protherm])
        spearman_all = spearmanr(y_all, pred_all)[0]
        results["combined"] = {"spearman": spearman_all, "n": len(y_all)}

    return results


def main():
    """Train hybrid attention transformer over VAE + Transformer specialists."""
    print("=" * 70)
    print("HYBRID ATTENTION TRANSFORMER")
    print("Fusing VAE + Transformer specialists with cross-attention")
    print("=" * 70)

    config = HybridConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Find latest VAE suite output
    vae_suite_dirs = sorted(
        project_root.glob("outputs/full_vae_suite_*"), key=lambda x: x.stat().st_mtime, reverse=True
    )

    if not vae_suite_dirs:
        print("ERROR: No VAE suite outputs found. Run train_full_vae_suite.py first.")
        return

    vae_suite_dir = vae_suite_dirs[0]
    print(f"\nUsing VAE suite: {vae_suite_dir.name}")

    # Load checkpoints
    print("\nLoading specialist checkpoints...")
    checkpoints = load_checkpoints(vae_suite_dir)

    if not checkpoints:
        print("ERROR: No checkpoints found")
        return

    # Load data
    print("\nLoading combined data...")
    X, y, source, (X_s669, y_s669, X_protherm, y_protherm) = load_combined_data()

    input_dim = X.shape[1]
    print(f"Input dimension: {input_dim}")

    # Create specialists
    print("\nCreating specialist models...")
    specialists = create_specialists(config, input_dim)

    # Load weights
    print("\nLoading specialist weights...")
    load_specialist_weights(specialists, checkpoints)

    # Train S669 transformer from scratch (since we may not have a compatible checkpoint)
    print("\nTraining S669 Transformer from scratch (to compensate weak VAE)...")
    s669_transformer = specialists["s669_transformer"]

    # Quick training of S669 transformer
    s669_t_optimizer = torch.optim.AdamW(s669_transformer.parameters(), lr=1e-4)
    s669_transformer.to(device)
    s669_transformer.train()

    # Split S669 data
    n_s669 = len(X_s669)
    idx_s669 = np.random.permutation(n_s669)
    split = int(0.8 * n_s669)
    X_s669_train, X_s669_val = X_s669[idx_s669[:split]], X_s669[idx_s669[split:]]
    y_s669_train, y_s669_val = y_s669[idx_s669[:split]], y_s669[idx_s669[split:]]

    train_loader_s669 = DataLoader(
        TensorDataset(torch.tensor(X_s669_train, dtype=torch.float32), torch.tensor(y_s669_train, dtype=torch.float32)),
        batch_size=32,
        shuffle=True,
    )

    best_s669_t_spearman = -1
    best_s669_t_state = None

    for epoch in range(100):
        s669_transformer.train()
        for batch_x, batch_y in train_loader_s669:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            out = s669_transformer(batch_x)
            loss = F.mse_loss(out["pred"], batch_y)
            s669_t_optimizer.zero_grad()
            loss.backward()
            s669_t_optimizer.step()

        # Validate
        s669_transformer.eval()
        with torch.no_grad():
            X_val_t = torch.tensor(X_s669_val, dtype=torch.float32, device=device)
            val_out = s669_transformer(X_val_t)
            val_spearman = spearmanr(y_s669_val, val_out["pred"].cpu().numpy())[0]

        if val_spearman > best_s669_t_spearman:
            best_s669_t_spearman = val_spearman
            best_s669_t_state = s669_transformer.state_dict().copy()

        if epoch % 20 == 0:
            print(f"  S669 Transformer epoch {epoch}: spearman={val_spearman:.4f}")

    if best_s669_t_state:
        s669_transformer.load_state_dict(best_s669_t_state)
    print(f"  S669 Transformer trained: best spearman={best_s669_t_spearman:.4f}")

    # Calculate specialist feature dimensions
    with torch.no_grad():
        dummy_x = torch.randn(1, input_dim, device=device)

        # S669 VAE + Refiner features
        s669_vae_out = specialists["s669_vae"].to(device)(dummy_x)
        s669_refiner_out = specialists["s669_refiner"].to(device)(s669_vae_out["mu"], s669_vae_out["pred"])
        s669_vae_refiner_dim = (
            s669_vae_out["mu"].shape[-1]  # latent_dim
            + s669_refiner_out["hidden"].shape[-1]  # refiner hidden
            + 2  # two predictions
        )

        # S669 Transformer features
        s669_t_out = s669_transformer(dummy_x)
        s669_transformer_dim = s669_t_out["hidden"].shape[-1] + 1  # hidden + pred

        # ProTherm VAE + Refiner features
        protherm_vae_out = specialists["protherm_vae"].to(device)(dummy_x)
        protherm_refiner_out = specialists["protherm_refiner"].to(device)(
            protherm_vae_out["mu"], protherm_vae_out["pred"]
        )
        protherm_vae_refiner_dim = protherm_vae_out["mu"].shape[-1] + protherm_refiner_out["hidden"].shape[-1] + 2

    specialist_dims = {
        "s669_vae_refiner": s669_vae_refiner_dim,
        "s669_transformer": s669_transformer_dim,
        "protherm_vae_refiner": protherm_vae_refiner_dim,
    }

    print("\nSpecialist feature dimensions:")
    for name, dim in specialist_dims.items():
        print(f"  {name}: {dim}")

    # Create attention transformer
    print("\nCreating Hybrid Attention Transformer...")
    attention_model = HybridAttentionTransformer(config, specialist_dims)

    # Train/val split (stratified by source)
    np.random.seed(42)

    # Stratified split
    s669_idx = np.where(source == 0)[0]
    protherm_idx = np.where(source == 1)[0]

    np.random.shuffle(s669_idx)
    np.random.shuffle(protherm_idx)

    s669_split = int(0.8 * len(s669_idx))
    protherm_split = int(0.8 * len(protherm_idx))

    train_idx = np.concatenate([s669_idx[:s669_split], protherm_idx[:protherm_split]])
    val_idx = np.concatenate([s669_idx[s669_split:], protherm_idx[protherm_split:]])

    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    print(f"\nTraining split: {len(X_train)} train, {len(X_val)} val")

    # Train
    print("\nTraining Hybrid Attention Transformer...")
    history = train_hybrid_attention(specialists, attention_model, X_train, y_train, X_val, y_val, config, device)

    print(f"\nTraining complete. Best Spearman: {history['best_spearman']:.4f}")

    # Evaluate by source
    print("\nEvaluating by dataset source...")
    results = evaluate_by_source(specialists, attention_model, X_s669, y_s669, X_protherm, y_protherm, config, device)

    print(f"""
Hybrid Attention Transformer Results:
=====================================
Combined (N={results["combined"]["n"]}): Spearman = {results["combined"]["spearman"]:.4f}
S669 (N={results["s669"]["n"]}):         Spearman = {results["s669"]["spearman"]:.4f}
ProTherm (N={results["protherm"]["n"]}): Spearman = {results["protherm"]["spearman"]:.4f}

Comparison with individual models:
----------------------------------
S669 VAE:             0.28 → Hybrid: {results["s669"]["spearman"]:.2f} ({"+" if results["s669"]["spearman"] > 0.28 else ""}{(results["s669"]["spearman"] - 0.28) / 0.28 * 100:.1f}%)
S669 Transformer:     {best_s669_t_spearman:.2f}
ProTherm VAE:         0.85 → Hybrid: {results["protherm"]["spearman"]:.2f} ({"+" if results["protherm"]["spearman"] > 0.85 else ""}{(results["protherm"]["spearman"] - 0.85) / 0.85 * 100:.1f}%)
ProTherm Refiner:     0.89 (BEST individual)
""")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "outputs" / f"hybrid_attention_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": attention_model.state_dict(),
            "config": config.__dict__,
            "specialist_dims": specialist_dims,
            "history": history,
            "results": results,
            "s669_transformer_spearman": best_s669_t_spearman,
        },
        output_dir / "model.pt",
    )

    # Save S669 transformer separately
    torch.save(
        {
            "model_state_dict": s669_transformer.state_dict(),
            "best_spearman": best_s669_t_spearman,
        },
        output_dir / "s669_transformer.pt",
    )

    with open(output_dir / "results.json", "w") as f:
        json.dump(
            {
                "combined_spearman": results["combined"]["spearman"],
                "s669_spearman": results["s669"]["spearman"],
                "protherm_spearman": results["protherm"]["spearman"],
                "s669_transformer_spearman": best_s669_t_spearman,
                "training_history": history,
            },
            f,
            indent=2,
        )

    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
