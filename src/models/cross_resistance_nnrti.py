"""Cross-Resistance VAE for NNRTI Drugs.

NNRTI drugs bind to the same hydrophobic pocket in RT, leading to
significant cross-resistance between first-generation NNRTIs (EFV, NVP)
and variable cross-resistance with second-generation (ETR, RPV, DOR).

Key patterns:
- K103N: Major mutation affecting EFV/NVP but not ETR/RPV
- Y181C: Affects all NNRTIs
- ETR/RPV share mutations at 100, 138, 179, 181

References:
- Vingerhoets et al., 2010: ETR resistance
- Azijn et al., 2010: RPV cross-resistance
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

# NNRTI Cross-Resistance Matrix
NNRTI_CROSS_RESISTANCE = {
    "NVP": {"NVP": 1.0, "EFV": 0.85, "ETR": 0.45, "RPV": 0.50, "DOR": 0.40},
    "EFV": {"NVP": 0.85, "EFV": 1.0, "ETR": 0.50, "RPV": 0.55, "DOR": 0.45},
    "ETR": {"NVP": 0.45, "EFV": 0.50, "ETR": 1.0, "RPV": 0.75, "DOR": 0.65},
    "RPV": {"NVP": 0.50, "EFV": 0.55, "ETR": 0.75, "RPV": 1.0, "DOR": 0.60},
    "DOR": {"NVP": 0.40, "EFV": 0.45, "ETR": 0.65, "RPV": 0.60, "DOR": 1.0},
}

# Key NNRTI mutation positions
NNRTI_MUTATIONS = {
    "first_gen": [100, 101, 103, 106, 188, 190],  # Affect NVP/EFV
    "second_gen": [100, 138, 179, 181, 227, 230],  # Affect ETR/RPV
    "shared": [100, 101, 181],  # Affect all NNRTIs
}


@dataclass
class NNRTIConfig:
    """Configuration for NNRTI cross-resistance VAE."""

    input_dim: int
    latent_dim: int = 16
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    drug_names: list[str] = field(default_factory=lambda: ["NVP", "EFV", "ETR", "RPV", "DOR"])
    n_positions: int = 318
    dropout: float = 0.1
    n_heads: int = 4
    ranking_weight: float = 0.3


class GenerationSpecificEncoder(nn.Module):
    """Encode mutation patterns specific to first vs second generation NNRTIs."""

    def __init__(self, n_positions: int = 318, embed_dim: int = 32):
        super().__init__()
        self.first_gen_pos = NNRTI_MUTATIONS["first_gen"]
        self.second_gen_pos = NNRTI_MUTATIONS["second_gen"]

        self.first_gen_embed = nn.Linear(len(self.first_gen_pos), embed_dim)
        self.second_gen_embed = nn.Linear(len(self.second_gen_pos), embed_dim)
        self.combine = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, x: torch.Tensor) -> tuple:
        """Extract generation-specific features.

        Returns:
            first_gen_features, second_gen_features, combined
        """
        batch_size = x.size(0)
        device = x.device
        n_aa = 22

        x_reshaped = x.view(batch_size, -1, n_aa)

        # First generation features
        first_gen_features = []
        for pos in self.first_gen_pos:
            if pos <= x_reshaped.size(1):
                pos_data = x_reshaped[:, pos - 1, :]
                is_mutated = (pos_data[:, :-1].sum(dim=-1) > 0).float()
                first_gen_features.append(is_mutated)
        first_gen = (
            torch.stack(first_gen_features, dim=-1)
            if first_gen_features
            else torch.zeros(batch_size, len(self.first_gen_pos), device=device)
        )

        # Second generation features
        second_gen_features = []
        for pos in self.second_gen_pos:
            if pos <= x_reshaped.size(1):
                pos_data = x_reshaped[:, pos - 1, :]
                is_mutated = (pos_data[:, :-1].sum(dim=-1) > 0).float()
                second_gen_features.append(is_mutated)
        second_gen = (
            torch.stack(second_gen_features, dim=-1)
            if second_gen_features
            else torch.zeros(batch_size, len(self.second_gen_pos), device=device)
        )

        # Embed
        first_embed = self.first_gen_embed(first_gen)
        second_embed = self.second_gen_embed(second_gen)

        combined = self.combine(torch.cat([first_embed, second_embed], dim=-1))

        return first_embed, second_embed, combined


class NNRTICrossResistanceVAE(nn.Module):
    """VAE with cross-resistance modeling for NNRTI drugs."""

    def __init__(self, cfg: NNRTIConfig):
        super().__init__()
        self.cfg = cfg
        self.drug_names = cfg.drug_names
        self.n_drugs = len(cfg.drug_names)

        # First/second generation classification
        self.first_gen_drugs = {"NVP", "EFV"}
        self.second_gen_drugs = {"ETR", "RPV", "DOR"}

        # Generation-specific encoder
        self.gen_encoder = GenerationSpecificEncoder(cfg.n_positions, 32)

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

        # Drug embeddings with generation info
        self.drug_embed = nn.Embedding(self.n_drugs, cfg.latent_dim)

        # Cross-drug attention
        self.cross_attention = nn.MultiheadAttention(cfg.latent_dim, cfg.n_heads, batch_first=True)

        # Drug-specific heads with generation-specific features
        self.drug_heads = nn.ModuleDict()
        for drug in cfg.drug_names:
            # Use first or second gen features
            gen_dim = 32  # Size of generation embedding
            self.drug_heads[drug] = nn.Sequential(
                nn.Linear(cfg.latent_dim + gen_dim, 32),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(32, 1),
            )

        # Cross-resistance matrix
        self.register_buffer("cross_resistance_matrix", self._build_matrix())

    def _build_matrix(self) -> torch.Tensor:
        n = len(self.drug_names)
        matrix = torch.zeros(n, n)
        for i, d1 in enumerate(self.drug_names):
            for j, d2 in enumerate(self.drug_names):
                matrix[i, j] = NNRTI_CROSS_RESISTANCE.get(d1, {}).get(d2, 0)
        return matrix

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = x.size(0)
        device = x.device

        # Generation-specific features
        first_gen_feat, second_gen_feat, gen_combined = self.gen_encoder(x)

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

        # Add latent
        z_expanded = z.unsqueeze(1).expand(-1, self.n_drugs, -1)
        query = drug_embeds + z_expanded

        # Cross-drug attention
        z_drugs, attn_weights = self.cross_attention(query, query, query)

        # Predict for each drug
        predictions = {}
        for i, drug in enumerate(self.drug_names):
            z_drug = z_drugs[:, i, :]

            # Use appropriate generation features
            gen_feat = first_gen_feat if drug in self.first_gen_drugs else second_gen_feat

            z_with_gen = torch.cat([z_drug, gen_feat], dim=-1)
            pred = self.drug_heads[drug](z_with_gen).squeeze(-1)
            predictions[drug] = pred

        return {
            "x_recon": x_recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "predictions": predictions,
            "first_gen_features": first_gen_feat,
            "second_gen_features": second_gen_feat,
            "attention_weights": attn_weights,
        }


if __name__ == "__main__":
    print("Testing NNRTI Cross-Resistance VAE")
    print("=" * 60)

    cfg = NNRTIConfig(input_dim=318 * 22)
    model = NNRTICrossResistanceVAE(cfg)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    x = torch.randn(4, 318 * 22)
    out = model(x)

    print(f"\nOutput keys: {list(out.keys())}")
    print(f"Predictions for: {list(out['predictions'].keys())}")
    print(f"First-gen features shape: {out['first_gen_features'].shape}")
    print(f"Second-gen features shape: {out['second_gen_features'].shape}")

    print("\nCross-resistance matrix:")
    for d1 in ["NVP", "EFV", "ETR", "RPV", "DOR"]:
        row = [f"{NNRTI_CROSS_RESISTANCE[d1][d2]:.2f}" for d2 in ["NVP", "EFV", "ETR", "RPV", "DOR"]]
        print(f"  {d1}: {' '.join(row)}")

    print("\n" + "=" * 60)
    print("NNRTI Cross-Resistance VAE working!")
