"""Ensemble Model: Best Architecture per Drug Class.

Combines the optimal architecture for each drug class:
- PI: Standard VAE (already optimal with abundant data)
- NRTI: Attention VAE (best for TAM patterns)
- NNRTI: Transformer VAE (best for low-data scenarios)
- INI: Transformer VAE (best for low-data scenarios)

Usage:
    ensemble = DrugClassEnsemble()
    pred = ensemble.predict(x, drug_class="nrti", drug="AZT")
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EnsembleConfig:
    """Configuration for ensemble model."""

    latent_dim: int = 16
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.1
    n_heads: int = 4
    transformer_layers: int = 2
    ranking_weight: float = 0.3


class StandardVAEModule(nn.Module):
    """Standard VAE for PI drugs."""

    def __init__(self, input_dim: int, cfg: EnsembleConfig):
        super().__init__()

        layers = []
        in_dim = input_dim
        for h in cfg.hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, h),
                    nn.GELU(),
                    nn.LayerNorm(h),
                    nn.Dropout(cfg.dropout),
                ]
            )
            in_dim = h
        self.encoder = nn.Sequential(*layers)

        self.fc_mu = nn.Linear(in_dim, cfg.latent_dim)
        self.fc_logvar = nn.Linear(in_dim, cfg.latent_dim)

        dec_layers = []
        dec_in = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            dec_layers.extend([nn.Linear(dec_in, h), nn.GELU()])
            dec_in = h
        dec_layers.append(nn.Linear(dec_in, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        self.predictor = nn.Sequential(
            nn.Linear(cfg.latent_dim, 32),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        x_recon = self.decoder(z)
        pred = self.predictor(z).squeeze(-1)
        return {"x_recon": x_recon, "mu": mu, "logvar": logvar, "z": z, "prediction": pred}


class AttentionVAEModule(nn.Module):
    """Attention VAE for NRTI drugs (TAM patterns)."""

    def __init__(self, input_dim: int, n_positions: int, cfg: EnsembleConfig):
        super().__init__()
        self.n_positions = n_positions
        self.n_aa = 22

        # Position-wise attention
        self.position_embed = nn.Embedding(n_positions, 64)
        self.attention = nn.MultiheadAttention(64, cfg.n_heads, batch_first=True)

        # Encoder after attention
        self.pre_encoder = nn.Linear(22, 64)
        layers = []
        in_dim = 64 * n_positions
        for h in cfg.hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, h),
                    nn.GELU(),
                    nn.LayerNorm(h),
                    nn.Dropout(cfg.dropout),
                ]
            )
            in_dim = h
        self.encoder = nn.Sequential(*layers)

        self.fc_mu = nn.Linear(in_dim, cfg.latent_dim)
        self.fc_logvar = nn.Linear(in_dim, cfg.latent_dim)

        # Decoder
        dec_layers = []
        dec_in = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            dec_layers.extend([nn.Linear(dec_in, h), nn.GELU()])
            dec_in = h
        dec_layers.append(nn.Linear(dec_in, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        self.predictor = nn.Sequential(
            nn.Linear(cfg.latent_dim, 32),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device

        # Reshape to (batch, n_positions, n_aa)
        x_reshaped = x.view(batch_size, self.n_positions, self.n_aa)

        # Position embeddings
        pos_idx = torch.arange(self.n_positions, device=device)
        pos_embed = self.position_embed(pos_idx).unsqueeze(0).expand(batch_size, -1, -1)

        # Project amino acids
        x_proj = self.pre_encoder(x_reshaped)

        # Add position embeddings and apply attention
        x_pos = x_proj + pos_embed
        attn_out, attn_weights = self.attention(x_pos, x_pos, x_pos)

        # Flatten and encode
        h = attn_out.reshape(batch_size, -1)
        h = self.encoder(h)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)

        x_recon = self.decoder(z)
        pred = self.predictor(z).squeeze(-1)

        return {
            "x_recon": x_recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "prediction": pred,
            "attention_weights": attn_weights,
        }


class TransformerVAEModule(nn.Module):
    """Transformer VAE for NNRTI/INI drugs (low-data scenarios)."""

    def __init__(self, input_dim: int, n_positions: int, cfg: EnsembleConfig):
        super().__init__()
        self.n_positions = n_positions
        self.n_aa = 22
        self.d_model = 64

        # Input projection
        self.input_proj = nn.Linear(22, self.d_model)
        self.pos_embed = nn.Embedding(n_positions, self.d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=128,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.transformer_layers)

        # Latent projection
        self.fc_mu = nn.Linear(self.d_model * n_positions, cfg.latent_dim)
        self.fc_logvar = nn.Linear(self.d_model * n_positions, cfg.latent_dim)

        # Decoder
        dec_layers = []
        dec_in = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            dec_layers.extend([nn.Linear(dec_in, h), nn.GELU()])
            dec_in = h
        dec_layers.append(nn.Linear(dec_in, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        self.predictor = nn.Sequential(
            nn.Linear(cfg.latent_dim, 32),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device

        # Reshape to (batch, n_positions, n_aa)
        x_reshaped = x.view(batch_size, self.n_positions, self.n_aa)

        # Project and add position embeddings
        x_proj = self.input_proj(x_reshaped)
        pos_idx = torch.arange(self.n_positions, device=device)
        pos_embed = self.pos_embed(pos_idx).unsqueeze(0).expand(batch_size, -1, -1)
        x_pos = x_proj + pos_embed

        # Transformer
        h = self.transformer(x_pos)

        # Flatten and project to latent
        h_flat = h.reshape(batch_size, -1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)

        x_recon = self.decoder(z)
        pred = self.predictor(z).squeeze(-1)

        return {"x_recon": x_recon, "mu": mu, "logvar": logvar, "z": z, "prediction": pred}


class DrugClassEnsemble(nn.Module):
    """Ensemble using best architecture per drug class."""

    DRUG_CLASS_MAPPING = {
        "pi": ["FPV", "ATV", "IDV", "LPV", "NFV", "SQV", "TPV", "DRV"],
        "nrti": ["ABC", "AZT", "D4T", "DDI", "3TC", "TDF"],
        "nnrti": ["DOR", "EFV", "ETR", "NVP", "RPV"],
        "ini": ["BIC", "DTG", "EVG", "RAL"],
    }

    POSITION_COUNTS = {
        "pi": 99,
        "nrti": 240,
        "nnrti": 318,
        "ini": 288,
    }

    def __init__(self, cfg: EnsembleConfig | None = None):
        super().__init__()
        self.cfg = cfg or EnsembleConfig()
        self.models: dict[str, nn.Module] = {}

    def get_model_for_class(self, drug_class: str, input_dim: int) -> nn.Module:
        """Get or create model for drug class."""
        if drug_class not in self.models:
            n_positions = self.POSITION_COUNTS.get(drug_class, 99)

            if drug_class == "pi":
                model = StandardVAEModule(input_dim, self.cfg)
            elif drug_class == "nrti":
                model = AttentionVAEModule(input_dim, n_positions, self.cfg)
            else:  # nnrti, ini
                model = TransformerVAEModule(input_dim, n_positions, self.cfg)

            self.models[drug_class] = model
            self.add_module(f"model_{drug_class}", model)

        return self.models[drug_class]

    def get_drug_class(self, drug: str) -> str:
        """Get drug class for a specific drug."""
        for drug_class, drugs in self.DRUG_CLASS_MAPPING.items():
            if drug in drugs:
                return drug_class
        raise ValueError(f"Unknown drug: {drug}")

    def forward(self, x: torch.Tensor, drug_class: str) -> dict[str, torch.Tensor]:
        """Forward pass for specific drug class."""
        model = self.get_model_for_class(drug_class, x.size(1))
        return model(x)

    def predict(self, x: torch.Tensor, drug: str) -> torch.Tensor:
        """Predict resistance for a specific drug."""
        drug_class = self.get_drug_class(drug)
        model = self.get_model_for_class(drug_class, x.size(1))
        model.eval()
        with torch.no_grad():
            out = model(x)
        return out["prediction"]


def compute_ensemble_loss(
    cfg: EnsembleConfig,
    out: dict[str, torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute loss for ensemble model."""
    losses = {}

    # Reconstruction
    losses["recon"] = F.mse_loss(out["x_recon"], x)

    # KL divergence
    kl = -0.5 * torch.mean(1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp())
    losses["kl"] = 0.001 * kl

    # Ranking loss
    pred = out["prediction"]
    p_c = pred - pred.mean()
    y_c = y - y.mean()
    p_std = torch.sqrt(torch.sum(p_c**2) + 1e-8)
    y_std = torch.sqrt(torch.sum(y_c**2) + 1e-8)
    corr = torch.sum(p_c * y_c) / (p_std * y_std)
    losses["ranking"] = cfg.ranking_weight * (-corr)

    losses["total"] = losses["recon"] + losses["kl"] + losses["ranking"]

    return losses


if __name__ == "__main__":
    print("Testing Drug Class Ensemble")
    print("=" * 60)

    cfg = EnsembleConfig()
    ensemble = DrugClassEnsemble(cfg)

    # Test PI (Standard VAE)
    x_pi = torch.randn(4, 99 * 22)
    out = ensemble.forward(x_pi, "pi")
    print(f"PI prediction shape: {out['prediction'].shape}")

    # Test NRTI (Attention VAE)
    x_nrti = torch.randn(4, 240 * 22)
    out = ensemble.forward(x_nrti, "nrti")
    print(f"NRTI prediction shape: {out['prediction'].shape}")
    print(f"NRTI attention weights: {out.get('attention_weights', 'N/A')}")

    # Test NNRTI (Transformer VAE)
    x_nnrti = torch.randn(4, 318 * 22)
    out = ensemble.forward(x_nnrti, "nnrti")
    print(f"NNRTI prediction shape: {out['prediction'].shape}")

    # Test drug lookup
    for drug in ["LPV", "AZT", "EFV", "RAL"]:
        drug_class = ensemble.get_drug_class(drug)
        print(f"{drug} -> {drug_class}")

    print("\n" + "=" * 60)
    print("Ensemble working!")
