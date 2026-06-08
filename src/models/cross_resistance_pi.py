"""Cross-Resistance VAE for PI Drugs.

PI drugs share overlapping resistance mutations at the protease active site.
Key cross-resistance patterns:
- LPV/DRV share many major mutations (32, 47, 50, 54, 76, 84)
- TPV has unique mutations (33, 58, 83)
- DRV has highest barrier to resistance

References:
- Wensing et al., 2019: IAS-USA drug resistance mutations
- Stanford HIVDB: PI genotypic scoring
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

# PI Cross-Resistance Matrix (based on Stanford HIVDB overlap)
PI_CROSS_RESISTANCE = {
    "LPV": {"LPV": 1.0, "ATV": 0.75, "FPV": 0.80, "IDV": 0.70, "NFV": 0.65, "SQV": 0.60, "TPV": 0.40, "DRV": 0.70},
    "ATV": {"LPV": 0.75, "ATV": 1.0, "FPV": 0.70, "IDV": 0.75, "NFV": 0.60, "SQV": 0.65, "TPV": 0.35, "DRV": 0.60},
    "FPV": {"LPV": 0.80, "ATV": 0.70, "FPV": 1.0, "IDV": 0.65, "NFV": 0.55, "SQV": 0.55, "TPV": 0.40, "DRV": 0.65},
    "IDV": {"LPV": 0.70, "ATV": 0.75, "FPV": 0.65, "IDV": 1.0, "NFV": 0.60, "SQV": 0.70, "TPV": 0.35, "DRV": 0.55},
    "NFV": {"LPV": 0.65, "ATV": 0.60, "FPV": 0.55, "IDV": 0.60, "NFV": 1.0, "SQV": 0.55, "TPV": 0.35, "DRV": 0.50},
    "SQV": {"LPV": 0.60, "ATV": 0.65, "FPV": 0.55, "IDV": 0.70, "NFV": 0.55, "SQV": 1.0, "TPV": 0.30, "DRV": 0.50},
    "TPV": {"LPV": 0.40, "ATV": 0.35, "FPV": 0.40, "IDV": 0.35, "NFV": 0.35, "SQV": 0.30, "TPV": 1.0, "DRV": 0.45},
    "DRV": {"LPV": 0.70, "ATV": 0.60, "FPV": 0.65, "IDV": 0.55, "NFV": 0.50, "SQV": 0.50, "TPV": 0.45, "DRV": 1.0},
}

# Key mutation positions in protease (1-99)
PI_KEY_MUTATIONS = {
    "major": [30, 32, 33, 46, 47, 48, 50, 54, 76, 82, 84, 88, 90],
    "accessory": [10, 11, 16, 20, 24, 35, 36, 53, 62, 63, 71, 73, 74, 77, 85, 93],
}


@dataclass
class PIConfig:
    """Configuration for PI cross-resistance VAE."""

    input_dim: int
    latent_dim: int = 16
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    drug_names: list[str] = field(default_factory=lambda: ["LPV", "ATV", "FPV", "IDV", "NFV", "SQV", "TPV", "DRV"])
    n_positions: int = 99
    dropout: float = 0.1
    ranking_weight: float = 0.3


class MutationEncoder(nn.Module):
    """Encode known mutation positions with learned embeddings."""

    def __init__(self, n_positions: int = 99, embed_dim: int = 32):
        super().__init__()
        self.major_positions = PI_KEY_MUTATIONS["major"]
        self.accessory_positions = PI_KEY_MUTATIONS["accessory"]

        # Learned embeddings for major vs accessory
        self.major_embed = nn.Embedding(len(self.major_positions), embed_dim)
        self.accessory_embed = nn.Embedding(len(self.accessory_positions), embed_dim)

        # Combine
        self.combine = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract mutation-aware embeddings.

        Args:
            x: One-hot encoded (batch, n_positions * 22)

        Returns:
            Mutation embedding (batch, embed_dim)
        """
        batch_size = x.size(0)
        device = x.device
        n_aa = 22

        # Reshape to (batch, n_positions, n_aa)
        x_reshaped = x.view(batch_size, -1, n_aa)

        # Check major positions
        major_features = []
        for i, pos in enumerate(self.major_positions):
            if pos <= x_reshaped.size(1):
                pos_data = x_reshaped[:, pos - 1, :]  # 0-indexed
                is_mutated = pos_data[:, :-1].sum(dim=-1) > 0  # Exclude gap
                embed = self.major_embed(torch.tensor([i], device=device))
                major_features.append(is_mutated.float().unsqueeze(1) * embed)

        # Check accessory positions
        accessory_features = []
        for i, pos in enumerate(self.accessory_positions):
            if pos <= x_reshaped.size(1):
                pos_data = x_reshaped[:, pos - 1, :]
                is_mutated = pos_data[:, :-1].sum(dim=-1) > 0
                embed = self.accessory_embed(torch.tensor([i], device=device))
                accessory_features.append(is_mutated.float().unsqueeze(1) * embed)

        if major_features:
            major_sum = torch.stack(major_features, dim=1).sum(dim=1)
        else:
            major_sum = torch.zeros(batch_size, 32, device=device)

        if accessory_features:
            accessory_sum = torch.stack(accessory_features, dim=1).sum(dim=1)
        else:
            accessory_sum = torch.zeros(batch_size, 32, device=device)

        combined = torch.cat([major_sum.squeeze(1), accessory_sum.squeeze(1)], dim=-1)
        return self.combine(combined)


class PICrossResistanceVAE(nn.Module):
    """VAE with cross-resistance modeling for PI drugs."""

    def __init__(self, cfg: PIConfig):
        super().__init__()
        self.cfg = cfg
        self.drug_names = cfg.drug_names
        self.n_drugs = len(cfg.drug_names)

        # Mutation encoder
        self.mutation_encoder = MutationEncoder(cfg.n_positions, 32)

        # Shared encoder
        layers = []
        in_dim = cfg.input_dim
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
        dec_layers.append(nn.Linear(dec_in, cfg.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        # Drug embeddings
        self.drug_embed = nn.Embedding(self.n_drugs, cfg.latent_dim)

        # Cross-drug attention
        self.cross_attention = nn.MultiheadAttention(cfg.latent_dim, 4, batch_first=True)

        # Drug-specific heads (latent + mutation embedding)
        self.drug_heads = nn.ModuleDict(
            {
                drug: nn.Sequential(
                    nn.Linear(cfg.latent_dim + 32, 32),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(32, 1),
                )
                for drug in cfg.drug_names
            }
        )

        # Cross-resistance matrix
        self.register_buffer("cross_resistance_matrix", self._build_matrix())

    def _build_matrix(self) -> torch.Tensor:
        n = len(self.drug_names)
        matrix = torch.zeros(n, n)
        for i, d1 in enumerate(self.drug_names):
            for j, d2 in enumerate(self.drug_names):
                matrix[i, j] = PI_CROSS_RESISTANCE.get(d1, {}).get(d2, 0)
        return matrix

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = x.size(0)
        device = x.device

        # Mutation features
        mut_features = self.mutation_encoder(x)

        # Encode
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)

        # Decode
        x_recon = self.decoder(z)

        # Drug embeddings
        drug_idx = torch.arange(self.n_drugs, device=device)
        drug_embeds = self.drug_embed(drug_idx).unsqueeze(0).expand(batch_size, -1, -1)

        # Add latent to drug embeddings
        z_expanded = z.unsqueeze(1).expand(-1, self.n_drugs, -1)
        query = drug_embeds + z_expanded

        # Cross-drug attention
        z_drugs, _ = self.cross_attention(query, query, query)

        # Predict for each drug
        predictions = {}
        for i, drug in enumerate(self.drug_names):
            z_drug = z_drugs[:, i, :]
            z_with_mut = torch.cat([z_drug, mut_features], dim=-1)
            pred = self.drug_heads[drug](z_with_mut).squeeze(-1)
            predictions[drug] = pred

        return {
            "x_recon": x_recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "predictions": predictions,
            "mutation_features": mut_features,
        }


if __name__ == "__main__":
    print("Testing PI Cross-Resistance VAE")
    print("=" * 60)

    cfg = PIConfig(input_dim=99 * 22)
    model = PICrossResistanceVAE(cfg)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    x = torch.randn(4, 99 * 22)
    out = model(x)

    print(f"\nOutput keys: {list(out.keys())}")
    print(f"Predictions for: {list(out['predictions'].keys())}")
    print(f"Mutation features shape: {out['mutation_features'].shape}")

    print("\n" + "=" * 60)
    print("PI Cross-Resistance VAE working!")
