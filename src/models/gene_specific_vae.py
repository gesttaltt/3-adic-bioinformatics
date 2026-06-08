"""Gene-specific VAE architectures for different HIV genes.

Different genes require different architectures:
- Protease (PR): 99 AA - current architecture works well
- Reverse Transcriptase (RT): 560 AA - needs deeper network + attention
- Integrase (IN): 288 AA - needs medium-sized network

This module provides gene-optimized VAE configurations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GeneConfig:
    """Configuration for gene-specific VAE."""

    gene: str  # "PR", "RT", "IN"
    input_dim: int
    latent_dim: int = 16
    hidden_dims: list[int] = field(default_factory=list)
    use_attention: bool = False
    attention_heads: int = 4
    dropout: float = 0.1
    use_residual: bool = False
    use_layer_norm: bool = True
    n_positions: int = 99  # Number of amino acid positions

    @classmethod
    def for_protease(cls) -> GeneConfig:
        """Optimized config for Protease (99 AA)."""
        return cls(
            gene="PR",
            input_dim=99 * 22,  # 2,178 features
            latent_dim=16,
            hidden_dims=[128, 64, 32],
            use_attention=False,
            dropout=0.1,
            n_positions=99,
        )

    @classmethod
    def for_reverse_transcriptase(cls) -> GeneConfig:
        """Optimized config for Reverse Transcriptase (560 AA)."""
        return cls(
            gene="RT",
            input_dim=560 * 22,  # 12,320 features
            latent_dim=32,
            hidden_dims=[512, 256, 128, 64],
            use_attention=True,
            attention_heads=8,
            dropout=0.2,
            use_residual=True,
            use_layer_norm=True,
            n_positions=560,
        )

    @classmethod
    def for_integrase(cls) -> GeneConfig:
        """Optimized config for Integrase (288 AA)."""
        return cls(
            gene="IN",
            input_dim=288 * 22,  # 6,336 features
            latent_dim=24,
            hidden_dims=[256, 128, 64],
            use_attention=True,
            attention_heads=4,
            dropout=0.15,
            use_residual=True,
            n_positions=288,
        )


class PositionalEncoding(nn.Module):
    """Positional encoding for sequence positions."""

    def __init__(self, d_model: int, max_len: int = 600, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class PositionAttention(nn.Module):
    """Multi-head attention over sequence positions."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len, d_model)
        attn_out, attn_weights = self.attention(x, x, x)
        x = self.norm(x + self.dropout(attn_out))
        return x, attn_weights


class ResidualBlock(nn.Module):
    """Residual block with optional layer norm."""

    def __init__(self, dim: int, dropout: float = 0.1, use_layer_norm: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim) if use_layer_norm else nn.Identity()
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.norm(x + residual)
        return x


class GeneSpecificEncoder(nn.Module):
    """Encoder with optional attention for long sequences."""

    def __init__(self, cfg: GeneConfig):
        super().__init__()
        self.cfg = cfg
        self.n_aa = 22  # Amino acid alphabet size

        if cfg.use_attention:
            # For attention: reshape to (batch, n_positions, n_aa) then project
            self.position_embed = nn.Linear(self.n_aa, cfg.hidden_dims[0])
            self.pos_encoding = PositionalEncoding(cfg.hidden_dims[0], cfg.n_positions, cfg.dropout)
            self.attention = PositionAttention(cfg.hidden_dims[0], cfg.attention_heads, cfg.dropout)

            # After attention, flatten and continue with MLP
            self.post_attention = nn.Linear(cfg.n_positions * cfg.hidden_dims[0], cfg.hidden_dims[0])

            # Build remaining encoder layers
            layers = []
            in_dim = cfg.hidden_dims[0]
            for h in cfg.hidden_dims[1:]:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.GELU())
                if cfg.use_layer_norm:
                    layers.append(nn.LayerNorm(h))
                layers.append(nn.Dropout(cfg.dropout))
                if cfg.use_residual and in_dim == h:
                    # Add residual connection placeholder
                    pass
                in_dim = h
            self.mlp = nn.Sequential(*layers)
            self.final_dim = cfg.hidden_dims[-1]
        else:
            # Standard MLP encoder
            layers = []
            in_dim = cfg.input_dim
            for h in cfg.hidden_dims:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.GELU())
                layers.append(nn.BatchNorm1d(h))
                layers.append(nn.Dropout(cfg.dropout))
                in_dim = h
            self.mlp = nn.Sequential(*layers)
            self.final_dim = cfg.hidden_dims[-1]
            self.attention = None

        self.fc_mu = nn.Linear(self.final_dim, cfg.latent_dim)
        self.fc_logvar = nn.Linear(self.final_dim, cfg.latent_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = x.size(0)
        attn_weights = None

        if self.cfg.use_attention:
            # Reshape: (batch, n_positions * n_aa) -> (batch, n_positions, n_aa)
            x = x.view(batch_size, self.cfg.n_positions, self.n_aa)

            # Project each position
            x = self.position_embed(x)  # (batch, n_positions, hidden_dim)
            x = self.pos_encoding(x)

            # Apply attention
            x, attn_weights = self.attention(x)

            # Flatten and continue
            x = x.view(batch_size, -1)
            x = self.post_attention(x)
            x = F.gelu(x)

        # MLP layers
        h = self.mlp(x)

        # Latent parameters
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return {"mu": mu, "logvar": logvar, "h": h, "attn_weights": attn_weights}


class GeneSpecificDecoder(nn.Module):
    """Decoder matching the encoder architecture."""

    def __init__(self, cfg: GeneConfig):
        super().__init__()
        self.cfg = cfg

        # Build decoder layers (reverse of encoder)
        layers = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.GELU())
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(cfg.dropout))
            in_dim = h

        layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class GeneSpecificVAE(nn.Module):
    """VAE optimized for specific HIV gene."""

    def __init__(self, cfg: GeneConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = GeneSpecificEncoder(cfg)
        self.decoder = GeneSpecificDecoder(cfg)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        enc_out = self.encoder(x)
        z = self.reparameterize(enc_out["mu"], enc_out["logvar"])
        x_recon = self.decoder(z)

        return {
            "x_recon": x_recon,
            "mu": enc_out["mu"],
            "logvar": enc_out["logvar"],
            "z": z,
            "attn_weights": enc_out.get("attn_weights"),
        }

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space (using mean)."""
        enc_out = self.encoder(x)
        return enc_out["mu"]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space."""
        return self.decoder(z)


class ConvolutionalEncoder(nn.Module):
    """1D CNN encoder for sequence data - alternative to attention."""

    def __init__(self, cfg: GeneConfig):
        super().__init__()
        self.cfg = cfg
        self.n_aa = 22

        # Conv layers over positions
        self.conv1 = nn.Conv1d(self.n_aa, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)

        self.pool = nn.AdaptiveAvgPool1d(16)  # Reduce to fixed size
        self.flatten_dim = 256 * 16

        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, cfg.hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

        # Build remaining layers
        layers = []
        in_dim = cfg.hidden_dims[0]
        for h in cfg.hidden_dims[1:]:
            layers.extend([nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(cfg.dropout)])
            in_dim = h
        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()

        self.fc_mu = nn.Linear(cfg.hidden_dims[-1], cfg.latent_dim)
        self.fc_logvar = nn.Linear(cfg.hidden_dims[-1], cfg.latent_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = x.size(0)

        # Reshape: (batch, n_positions * n_aa) -> (batch, n_aa, n_positions)
        x = x.view(batch_size, self.cfg.n_positions, self.n_aa)
        x = x.permute(0, 2, 1)  # (batch, n_aa, n_positions)

        # Conv layers
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))

        # Pool and flatten
        x = self.pool(x)
        x = x.view(batch_size, -1)

        # FC layers
        x = self.fc(x)
        h = self.mlp(x)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return {"mu": mu, "logvar": logvar, "h": h, "attn_weights": None}


class GeneSpecificCNNVAE(nn.Module):
    """VAE with CNN encoder for long sequences."""

    def __init__(self, cfg: GeneConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = ConvolutionalEncoder(cfg)
        self.decoder = GeneSpecificDecoder(cfg)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        enc_out = self.encoder(x)
        z = self.reparameterize(enc_out["mu"], enc_out["logvar"])
        x_recon = self.decoder(z)

        return {
            "x_recon": x_recon,
            "mu": enc_out["mu"],
            "logvar": enc_out["logvar"],
            "z": z,
            "attn_weights": None,
        }


def create_vae_for_gene(gene: str) -> GeneSpecificVAE:
    """Factory function to create appropriate VAE for gene type."""
    gene = gene.upper()

    if gene in ["PR", "PROTEASE"]:
        cfg = GeneConfig.for_protease()
    elif gene in ["RT", "REVERSE_TRANSCRIPTASE"]:
        cfg = GeneConfig.for_reverse_transcriptase()
    elif gene in ["IN", "INTEGRASE"]:
        cfg = GeneConfig.for_integrase()
    else:
        raise ValueError(f"Unknown gene: {gene}. Use PR, RT, or IN.")

    return GeneSpecificVAE(cfg)


def create_cnn_vae_for_gene(gene: str) -> GeneSpecificCNNVAE:
    """Factory function to create CNN VAE for gene type."""
    gene = gene.upper()

    if gene in ["PR", "PROTEASE"]:
        cfg = GeneConfig.for_protease()
    elif gene in ["RT", "REVERSE_TRANSCRIPTASE"]:
        cfg = GeneConfig.for_reverse_transcriptase()
    elif gene in ["IN", "INTEGRASE"]:
        cfg = GeneConfig.for_integrase()
    else:
        raise ValueError(f"Unknown gene: {gene}. Use PR, RT, or IN.")

    return GeneSpecificCNNVAE(cfg)


if __name__ == "__main__":
    # Test each configuration
    print("Testing Gene-Specific VAE Architectures")
    print("=" * 60)

    for gene, create_cfg in [
        ("PR", GeneConfig.for_protease),
        ("RT", GeneConfig.for_reverse_transcriptase),
        ("IN", GeneConfig.for_integrase),
    ]:
        cfg = create_cfg()
        print(f"\n{gene} Configuration:")
        print(f"  Input dim: {cfg.input_dim}")
        print(f"  Latent dim: {cfg.latent_dim}")
        print(f"  Hidden dims: {cfg.hidden_dims}")
        print(f"  Use attention: {cfg.use_attention}")

        # Create and test model
        model = GeneSpecificVAE(cfg)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        # Test forward pass
        x = torch.randn(8, cfg.input_dim)
        out = model(x)
        print(f"  Output shapes: z={out['z'].shape}, recon={out['x_recon'].shape}")

        if out["attn_weights"] is not None:
            print(f"  Attention shape: {out['attn_weights'].shape}")

    print("\n" + "=" * 60)
    print("All gene-specific architectures working!")
