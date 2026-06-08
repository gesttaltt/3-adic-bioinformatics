"""Subtype-Specific Drug Resistance Models.

HIV-1 subtypes (clades) can have different resistance patterns due to:
1. Baseline polymorphisms (natural variation)
2. Different mutation pathways
3. Geographic treatment patterns

This module provides:
1. Subtype-aware encoders
2. Multi-task learning across subtypes
3. Transfer learning for low-data subtypes
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

# Known HIV-1 subtypes and their characteristics
SUBTYPE_INFO = {
    "B": {"prevalence": "high", "region": "Americas/Europe", "data_abundance": "high"},
    "C": {"prevalence": "highest", "region": "Africa/India", "data_abundance": "medium"},
    "A": {"prevalence": "medium", "region": "Africa/Russia", "data_abundance": "medium"},
    "D": {"prevalence": "low", "region": "Africa", "data_abundance": "low"},
    "CRF01_AE": {"prevalence": "high", "region": "Asia", "data_abundance": "medium"},
    "CRF02_AG": {"prevalence": "medium", "region": "West Africa", "data_abundance": "low"},
    "G": {"prevalence": "low", "region": "Africa/Europe", "data_abundance": "low"},
}

# Known subtype-specific polymorphisms (baseline differences from B)
SUBTYPE_POLYMORPHISMS = {
    "C": {
        "PR": [12, 15, 19, 36, 37, 41, 63, 64, 69, 89, 93],
        "RT": [35, 122, 135, 162, 177, 200, 207, 211, 214, 245, 277, 286, 293],
        "IN": [17, 72, 101, 112, 119, 125, 201, 206, 215, 218, 234],
    },
    "A": {
        "PR": [12, 15, 19, 36, 37, 41, 63, 64, 89, 93],
        "RT": [35, 122, 135, 162, 207, 211, 245, 277, 286],
        "IN": [17, 72, 119, 125],
    },
    "CRF01_AE": {
        "PR": [13, 15, 19, 36, 37, 41, 63, 64, 69, 89, 93],
        "RT": [35, 122, 135, 162, 177, 200, 211, 214, 245, 277, 286, 293],
        "IN": [17, 72, 119, 201, 206],
    },
}


@dataclass
class SubtypeConfig:
    """Configuration for subtype-specific model."""

    input_dim: int
    latent_dim: int = 16
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    n_subtypes: int = 7
    subtype_embed_dim: int = 16
    dropout: float = 0.1
    share_encoder: bool = True
    use_polymorphism_encoding: bool = True


class SubtypeEncoder(nn.Module):
    """Encode subtype information with polymorphism awareness."""

    def __init__(self, n_subtypes: int = 7, embed_dim: int = 16):
        super().__init__()
        self.n_subtypes = n_subtypes
        self.embed_dim = embed_dim

        # Subtype names for indexing
        self.subtype_names = ["B", "C", "A", "D", "CRF01_AE", "CRF02_AG", "G"]
        self.subtype_to_idx = {s: i for i, s in enumerate(self.subtype_names)}

        # Learned subtype embeddings
        self.embedding = nn.Embedding(n_subtypes, embed_dim)

        # Polymorphism-aware MLP
        self.poly_encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, subtype_idx: torch.Tensor) -> torch.Tensor:
        """Get subtype embedding."""
        embed = self.embedding(subtype_idx)
        return self.poly_encoder(embed)

    def get_idx(self, subtype: str) -> int:
        """Convert subtype name to index."""
        return self.subtype_to_idx.get(subtype, 0)  # Default to B


class PolymorphismEncoder(nn.Module):
    """Encode known polymorphism positions for a subtype."""

    def __init__(self, n_positions: int, gene: str = "PR", embed_dim: int = 16):
        super().__init__()
        self.n_positions = n_positions
        self.gene = gene
        self.embed_dim = embed_dim

        # Get all polymorphism positions across subtypes
        all_positions = set()
        for subtype_data in SUBTYPE_POLYMORPHISMS.values():
            if gene in subtype_data:
                all_positions.update(subtype_data[gene])

        self.poly_positions = sorted(all_positions)
        self.n_poly = len(self.poly_positions)

        # Embedding for polymorphism patterns
        self.poly_embed = nn.Linear(self.n_poly, embed_dim) if self.n_poly > 0 else None

    def forward(self, x: torch.Tensor, subtype: str = "B") -> torch.Tensor:
        """Extract polymorphism pattern from sequence."""
        if self.poly_embed is None:
            return torch.zeros(x.size(0), self.embed_dim, device=x.device)

        batch_size = x.size(0)
        n_aa = 22

        # Reshape to (batch, n_positions, n_aa)
        x_reshaped = x.view(batch_size, -1, n_aa)

        # Get polymorphism pattern
        poly_pattern = []
        for pos in self.poly_positions:
            if pos <= x_reshaped.size(1):
                # Check if position differs from consensus (non-zero in any non-consensus position)
                pos_data = x_reshaped[:, pos - 1, :]
                is_polymorphic = pos_data.sum(dim=-1)  # Sum should be 1 if valid
                poly_pattern.append(is_polymorphic)

        if poly_pattern:
            poly_tensor = torch.stack(poly_pattern, dim=-1)
            return self.poly_embed(poly_tensor)
        else:
            return torch.zeros(batch_size, self.embed_dim, device=x.device)


class SubtypeSpecificVAE(nn.Module):
    """VAE with subtype-specific components."""

    def __init__(self, cfg: SubtypeConfig):
        super().__init__()
        self.cfg = cfg

        # Subtype encoder
        self.subtype_encoder = SubtypeEncoder(cfg.n_subtypes, cfg.subtype_embed_dim)

        # Polymorphism encoder (optional)
        if cfg.use_polymorphism_encoding:
            n_pos = cfg.input_dim // 22
            self.poly_encoder = PolymorphismEncoder(n_pos, "PR", cfg.subtype_embed_dim)
        else:
            self.poly_encoder = None

        # Shared encoder
        encoder_input = cfg.input_dim + cfg.subtype_embed_dim
        if cfg.use_polymorphism_encoding:
            encoder_input += cfg.subtype_embed_dim

        layers = []
        in_dim = encoder_input
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

        # Latent space
        self.fc_mu = nn.Linear(in_dim, cfg.latent_dim)
        self.fc_logvar = nn.Linear(in_dim, cfg.latent_dim)

        # Decoder
        dec_layers = []
        dec_in = cfg.latent_dim + cfg.subtype_embed_dim
        for h in reversed(cfg.hidden_dims):
            dec_layers.extend([nn.Linear(dec_in, h), nn.GELU()])
            dec_in = h
        dec_layers.append(nn.Linear(dec_in, cfg.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        # Predictor with subtype conditioning
        self.predictor = nn.Sequential(
            nn.Linear(cfg.latent_dim + cfg.subtype_embed_dim, 32),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        subtype_idx: torch.Tensor | None = None,
        subtype_name: str = "B",
    ) -> dict[str, torch.Tensor]:
        batch_size = x.size(0)
        device = x.device

        # Get subtype embedding
        if subtype_idx is None:
            idx = self.subtype_encoder.get_idx(subtype_name)
            subtype_idx = torch.full((batch_size,), idx, dtype=torch.long, device=device)

        subtype_embed = self.subtype_encoder(subtype_idx)

        # Get polymorphism encoding
        if self.poly_encoder is not None:
            poly_embed = self.poly_encoder(x, subtype_name)
            encoder_input = torch.cat([x, subtype_embed, poly_embed], dim=-1)
        else:
            encoder_input = torch.cat([x, subtype_embed], dim=-1)

        # Encode
        h = self.encoder(encoder_input)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)

        # Decode with subtype conditioning
        z_conditioned = torch.cat([z, subtype_embed], dim=-1)
        x_recon = self.decoder(z_conditioned)

        # Predict with subtype conditioning
        pred_input = torch.cat([z, subtype_embed], dim=-1)
        prediction = self.predictor(pred_input).squeeze(-1)

        return {
            "x_recon": x_recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "prediction": prediction,
            "subtype_embed": subtype_embed,
        }


class SubtypeTransferLearning(nn.Module):
    """Transfer learning for low-data subtypes.

    Strategy:
    1. Pre-train on high-data subtype (B)
    2. Fine-tune on target subtype with frozen encoder
    3. Gradually unfreeze layers
    """

    def __init__(self, base_model: SubtypeSpecificVAE):
        super().__init__()
        self.base_model = base_model

        # Subtype-specific adaptation layers
        self.adaptation_layers = nn.ModuleDict()

    def create_adapter(self, subtype: str, hidden_dim: int = 32):
        """Create adaptation layer for new subtype."""
        adapter = nn.Sequential(
            nn.Linear(self.base_model.cfg.latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.base_model.cfg.latent_dim),
        )
        self.adaptation_layers[subtype] = adapter
        return adapter

    def forward(
        self,
        x: torch.Tensor,
        subtype: str,
        use_adapter: bool = True,
    ) -> dict[str, torch.Tensor]:
        # Get base model output
        output = self.base_model(x, subtype_name=subtype)

        # Apply adapter if available
        if use_adapter and subtype in self.adaptation_layers:
            z_adapted = self.adaptation_layers[subtype](output["z"])
            output["z_adapted"] = z_adapted

            # Re-predict with adapted latent
            subtype_embed = output["subtype_embed"]
            pred_input = torch.cat([z_adapted, subtype_embed], dim=-1)
            output["prediction"] = self.base_model.predictor(pred_input).squeeze(-1)

        return output

    def freeze_encoder(self):
        """Freeze encoder for transfer learning."""
        for param in self.base_model.encoder.parameters():
            param.requires_grad = False
        for param in self.base_model.fc_mu.parameters():
            param.requires_grad = False
        for param in self.base_model.fc_logvar.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder for fine-tuning."""
        for param in self.base_model.encoder.parameters():
            param.requires_grad = True
        for param in self.base_model.fc_mu.parameters():
            param.requires_grad = True
        for param in self.base_model.fc_logvar.parameters():
            param.requires_grad = True


class MultiSubtypeVAE(nn.Module):
    """Multi-task VAE that jointly models all subtypes.

    Uses a shared encoder with subtype-specific prediction heads.
    Encourages learning of shared resistance patterns while allowing
    subtype-specific adaptations.
    """

    def __init__(self, cfg: SubtypeConfig):
        super().__init__()
        self.cfg = cfg
        self.subtypes = ["B", "C", "A", "D", "CRF01_AE", "CRF02_AG", "G"]

        # Shared encoder
        self.subtype_encoder = SubtypeEncoder(cfg.n_subtypes, cfg.subtype_embed_dim)

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
        self.shared_encoder = nn.Sequential(*layers)

        self.fc_mu = nn.Linear(in_dim, cfg.latent_dim)
        self.fc_logvar = nn.Linear(in_dim, cfg.latent_dim)

        # Subtype-specific prediction heads
        self.subtype_heads = nn.ModuleDict(
            {
                subtype: nn.Sequential(
                    nn.Linear(cfg.latent_dim + cfg.subtype_embed_dim, 32),
                    nn.GELU(),
                    nn.Linear(32, 1),
                )
                for subtype in self.subtypes
            }
        )

        # Shared prediction head (for regularization)
        self.shared_head = nn.Sequential(
            nn.Linear(cfg.latent_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        subtype: str = "B",
    ) -> dict[str, torch.Tensor]:
        batch_size = x.size(0)
        device = x.device

        # Get subtype embedding
        idx = self.subtype_encoder.get_idx(subtype)
        subtype_idx = torch.full((batch_size,), idx, dtype=torch.long, device=device)
        subtype_embed = self.subtype_encoder(subtype_idx)

        # Shared encoding
        h = self.shared_encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)

        # Subtype-specific prediction
        z_conditioned = torch.cat([z, subtype_embed], dim=-1)
        subtype_pred = self.subtype_heads[subtype](z_conditioned).squeeze(-1)

        # Shared prediction (for regularization)
        shared_pred = self.shared_head(z).squeeze(-1)

        return {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "prediction": subtype_pred,
            "shared_prediction": shared_pred,
            "subtype_embed": subtype_embed,
        }

    def predict_all_subtypes(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Get predictions for all subtypes."""
        batch_size = x.size(0)
        device = x.device

        # Shared encoding
        h = self.shared_encoder(x)
        mu = self.fc_mu(h)
        z = mu  # Use mean for prediction

        predictions = {}
        for subtype in self.subtypes:
            idx = self.subtype_encoder.get_idx(subtype)
            subtype_idx = torch.full((batch_size,), idx, dtype=torch.long, device=device)
            subtype_embed = self.subtype_encoder(subtype_idx)

            z_conditioned = torch.cat([z, subtype_embed], dim=-1)
            pred = self.subtype_heads[subtype](z_conditioned).squeeze(-1)
            predictions[subtype] = pred

        return predictions


if __name__ == "__main__":
    print("Testing Subtype-Specific Models")
    print("=" * 60)

    # Test SubtypeSpecificVAE
    cfg = SubtypeConfig(input_dim=99 * 22)
    model = SubtypeSpecificVAE(cfg)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"SubtypeSpecificVAE parameters: {n_params:,}")

    x = torch.randn(4, 99 * 22)
    out = model(x, subtype_name="C")

    print(f"Output keys: {list(out.keys())}")
    print(f"Prediction shape: {out['prediction'].shape}")
    print(f"Subtype embed shape: {out['subtype_embed'].shape}")

    # Test MultiSubtypeVAE
    print("\nTesting MultiSubtypeVAE...")
    multi_model = MultiSubtypeVAE(cfg)

    all_preds = multi_model.predict_all_subtypes(x)
    print(f"Predictions for subtypes: {list(all_preds.keys())}")

    # Test transfer learning
    print("\nTesting Transfer Learning...")
    transfer = SubtypeTransferLearning(model)
    transfer.create_adapter("D")
    transfer.freeze_encoder()

    out_transfer = transfer(x, subtype="D")
    print(f"Adapted prediction shape: {out_transfer['prediction'].shape}")

    print("\n" + "=" * 60)
    print("Subtype-specific models working!")
