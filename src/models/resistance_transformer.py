"""Transformer architecture for HIV drug resistance prediction.

Transformers have dominated NLP and are increasingly successful in
protein/sequence modeling. This module provides a transformer-based
alternative to the VAE approach for comparison.

Key differences from VAE:
- Attention over sequence positions
- Direct regression head (no latent space)
- Better handling of long-range dependencies

References:
- Vaswani et al. (2017): Attention Is All You Need
- Rives et al. (2021): Biological structure from protein language models
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    """Configuration for resistance transformer."""

    n_positions: int = 99  # Sequence length (99 for PR, 560 for RT)
    n_aa: int = 22  # Amino acid alphabet size
    d_model: int = 128  # Model dimension
    n_heads: int = 8  # Attention heads
    n_layers: int = 4  # Transformer layers
    d_ff: int = 512  # Feed-forward dimension
    dropout: float = 0.1
    max_seq_len: int = 600  # Maximum sequence length
    use_ranking_loss: bool = True
    ranking_weight: float = 0.3


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 600, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ResistanceTransformer(nn.Module):
    """Transformer for drug resistance prediction."""

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg

        # Amino acid embedding
        self.aa_embedding = nn.Linear(cfg.n_aa, cfg.d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(cfg.d_model, cfg.max_seq_len, cfg.dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)

        # Output head
        self.norm = nn.LayerNorm(cfg.d_model)
        self.fc = nn.Linear(cfg.d_model, 1)

        # For attention visualization
        self.attention_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: One-hot encoded sequence (batch, n_positions * n_aa)

        Returns:
            Dict with 'prediction' and 'attention'
        """
        batch_size = x.size(0)

        # Reshape to (batch, n_positions, n_aa)
        x = x.view(batch_size, self.cfg.n_positions, self.cfg.n_aa)

        # Embed amino acids
        x = self.aa_embedding(x)  # (batch, n_positions, d_model)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Transformer encoding
        x = self.transformer(x)  # (batch, n_positions, d_model)

        # Global average pooling
        x = x.mean(dim=1)  # (batch, d_model)

        # Normalize and predict
        x = self.norm(x)
        prediction = self.fc(x).squeeze(-1)  # (batch,)

        return {"prediction": prediction, "embedding": x}

    def get_attention_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract attention weights from all layers."""
        batch_size = x.size(0)
        x = x.view(batch_size, self.cfg.n_positions, self.cfg.n_aa)
        x = self.aa_embedding(x)
        x = self.pos_encoding(x)

        attention_maps = []
        for layer in self.transformer.layers:
            # Extract self-attention weights
            attn_output, attn_weights = layer.self_attn(x, x, x, need_weights=True)
            attention_maps.append(attn_weights.detach())
            x = layer(x)

        return attention_maps


class ResistanceTransformerWithVAE(nn.Module):
    """Transformer with VAE-style latent space for uncertainty."""

    def __init__(self, cfg: TransformerConfig, latent_dim: int = 16):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = latent_dim

        # Amino acid embedding
        self.aa_embedding = nn.Linear(cfg.n_aa, cfg.d_model)
        self.pos_encoding = PositionalEncoding(cfg.d_model, cfg.max_seq_len, cfg.dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)

        # VAE latent space
        self.fc_mu = nn.Linear(cfg.d_model, latent_dim)
        self.fc_logvar = nn.Linear(cfg.d_model, latent_dim)

        # Decoder (simple MLP from latent)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.n_positions * cfg.n_aa),
        )

        # Prediction head from latent
        self.predictor = nn.Linear(latent_dim, 1)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = x.size(0)

        # Reshape and embed
        x_reshaped = x.view(batch_size, self.cfg.n_positions, self.cfg.n_aa)
        h = self.aa_embedding(x_reshaped)
        h = self.pos_encoding(h)

        # Transformer encoding
        h = self.transformer(h)

        # Pool to single vector
        h = h.mean(dim=1)

        # VAE latent
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)

        # Reconstruction
        x_recon = self.decoder(z)

        # Prediction
        prediction = self.predictor(z).squeeze(-1)

        return {
            "x_recon": x_recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "prediction": prediction,
        }


def compute_transformer_loss(
    cfg: TransformerConfig,
    out: dict[str, torch.Tensor],
    x: torch.Tensor,
    fitness: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute losses for transformer training."""
    losses = {}

    prediction = out["prediction"]

    # MSE loss for direct prediction
    losses["mse"] = F.mse_loss(prediction, fitness)

    # Ranking loss (correlation)
    if cfg.use_ranking_loss:
        p_c = prediction - prediction.mean()
        f_c = fitness - fitness.mean()
        p_std = torch.sqrt(torch.sum(p_c**2) + 1e-8)
        f_std = torch.sqrt(torch.sum(f_c**2) + 1e-8)
        corr = torch.sum(p_c * f_c) / (p_std * f_std)
        losses["rank"] = cfg.ranking_weight * (-corr)

    # If VAE-style, add reconstruction and KL
    if "x_recon" in out:
        losses["recon"] = 0.1 * F.mse_loss(out["x_recon"], x)
        kl = -0.5 * torch.sum(1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp())
        losses["kl"] = 0.001 * kl / x.size(0)

    losses["total"] = sum(losses.values())
    return losses


class MultiHeadResistanceTransformer(nn.Module):
    """Transformer with multiple drug prediction heads (multi-task)."""

    def __init__(self, cfg: TransformerConfig, drug_names: list[str]):
        super().__init__()
        self.cfg = cfg
        self.drug_names = drug_names
        self.n_drugs = len(drug_names)

        # Shared encoder
        self.aa_embedding = nn.Linear(cfg.n_aa, cfg.d_model)
        self.pos_encoding = PositionalEncoding(cfg.d_model, cfg.max_seq_len, cfg.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)

        # Drug-specific heads
        self.drug_heads = nn.ModuleDict({drug: nn.Linear(cfg.d_model, 1) for drug in drug_names})

        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor, drug: str | None = None) -> dict[str, torch.Tensor]:
        batch_size = x.size(0)

        # Shared encoding
        x = x.view(batch_size, self.cfg.n_positions, self.cfg.n_aa)
        x = self.aa_embedding(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.norm(x)

        if drug is not None:
            # Single drug prediction
            return {"prediction": self.drug_heads[drug](x).squeeze(-1), "embedding": x}
        else:
            # All drug predictions
            predictions = {name: self.drug_heads[name](x).squeeze(-1) for name in self.drug_names}
            return {"predictions": predictions, "embedding": x}


def create_transformer_for_gene(gene: str, use_vae: bool = False) -> nn.Module:
    """Factory function to create transformer for specific gene."""
    gene = gene.upper()

    if gene in ["PR", "PROTEASE"]:
        cfg = TransformerConfig(n_positions=99, d_model=128, n_layers=4, n_heads=8)
    elif gene in ["RT", "REVERSE_TRANSCRIPTASE"]:
        cfg = TransformerConfig(n_positions=560, d_model=256, n_layers=6, n_heads=8)
    elif gene in ["IN", "INTEGRASE"]:
        cfg = TransformerConfig(n_positions=288, d_model=192, n_layers=5, n_heads=8)
    else:
        raise ValueError(f"Unknown gene: {gene}")

    if use_vae:
        return ResistanceTransformerWithVAE(cfg)
    else:
        return ResistanceTransformer(cfg)


if __name__ == "__main__":
    print("Testing Resistance Transformer")
    print("=" * 60)

    # Test basic transformer
    cfg = TransformerConfig(n_positions=99)
    model = ResistanceTransformer(cfg)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Test forward pass
    x = torch.randn(8, 99 * 22)
    out = model(x)
    print(f"Prediction shape: {out['prediction'].shape}")
    print(f"Embedding shape: {out['embedding'].shape}")

    # Test VAE variant
    model_vae = ResistanceTransformerWithVAE(cfg)
    out_vae = model_vae(x)
    print("\nVAE variant:")
    print(f"  z shape: {out_vae['z'].shape}")
    print(f"  prediction shape: {out_vae['prediction'].shape}")

    # Test multi-head
    drugs = ["LPV", "DRV", "ATV"]
    model_multi = MultiHeadResistanceTransformer(cfg, drugs)
    out_multi = model_multi(x)
    print("\nMulti-head variant:")
    for drug, pred in out_multi["predictions"].items():
        print(f"  {drug}: {pred.shape}")

    print("\n" + "=" * 60)
    print("All transformer variants working!")
