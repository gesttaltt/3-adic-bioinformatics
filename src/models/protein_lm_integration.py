"""Protein Language Model Integration for Drug Resistance Prediction.

Integrates pre-trained protein language models (ESM-2, ProtTrans) with our
p-adic VAE framework for improved generalization and few-shot learning.

Architecture:
1. Use ESM-2 to generate sequence embeddings
2. Project embeddings through adapter layers
3. Feed to VAE encoder (optionally freeze ESM)
4. Apply ranking loss for resistance prediction

Benefits:
- Pre-trained on 250M+ protein sequences
- Better generalization to novel mutations
- Improved performance on low-data drugs
- Transfer learning across pathogens

References:
- Lin et al., 2023: ESM-2 (Science)
- Elnaggar et al., 2021: ProtTrans (PAMI)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class PLMConfig:
    """Configuration for protein language model integration."""

    # PLM settings
    plm_name: str = "esm2_t6_8M_UR50D"  # Smallest ESM-2 for testing
    plm_dim: int = 320  # ESM-2 8M embedding dimension
    freeze_plm: bool = True
    use_mean_pooling: bool = True

    # Adapter settings
    adapter_dim: int = 128
    adapter_dropout: float = 0.1

    # VAE settings
    latent_dim: int = 16
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.1

    # Training
    ranking_weight: float = 0.3


class PLMAdapter(nn.Module):
    """Adapter layer to project PLM embeddings to VAE space."""

    def __init__(self, plm_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()

        self.down_project = nn.Linear(plm_dim, output_dim)
        self.activation = nn.GELU()
        self.up_project = nn.Linear(output_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project PLM embeddings."""
        residual = self.down_project(x)
        h = self.activation(residual)
        h = self.up_project(h)
        h = self.dropout(h)
        return self.layer_norm(h + residual)


class MockESM2(nn.Module):
    """Mock ESM-2 model for testing without installing fair-esm.

    Replace with actual ESM-2 in production:
    ```
    import esm
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    ```
    """

    def __init__(self, embed_dim: int = 320, num_layers: int = 6):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # Mock embedding layer
        self.embed_tokens = nn.Embedding(33, embed_dim)  # 33 = ESM alphabet size

        # Mock transformer layers
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=4,
                    dim_feedforward=embed_dim * 4,
                    dropout=0.1,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        tokens: torch.Tensor,
        repr_layers: list[int] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass mimicking ESM-2 output format."""
        x = self.embed_tokens(tokens)

        representations = {}
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if repr_layers is None or (i + 1) in repr_layers:
                representations[i + 1] = x

        return {
            "representations": representations,
            "logits": x,
        }


class PLMVAEEncoder(nn.Module):
    """VAE encoder that uses PLM embeddings as input."""

    def __init__(self, cfg: PLMConfig):
        super().__init__()
        self.cfg = cfg

        # PLM adapter
        self.adapter = PLMAdapter(cfg.plm_dim, cfg.adapter_dim, cfg.adapter_dropout)

        # Encoder layers
        layers = []
        in_dim = cfg.adapter_dim
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

        # Latent projections
        self.fc_mu = nn.Linear(in_dim, cfg.latent_dim)
        self.fc_logvar = nn.Linear(in_dim, cfg.latent_dim)

    def forward(self, plm_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode PLM embeddings to latent space.

        Args:
            plm_embeddings: (batch, seq_len, plm_dim) or (batch, plm_dim)
        """
        # Pool if sequence-level embeddings
        if plm_embeddings.dim() == 3:
            if self.cfg.use_mean_pooling:
                x = plm_embeddings.mean(dim=1)
            else:
                x = plm_embeddings[:, 0, :]  # CLS token
        else:
            x = plm_embeddings

        # Adapt and encode
        x = self.adapter(x)
        h = self.encoder(x)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar


class ProteinLMVAE(nn.Module):
    """VAE with Protein Language Model integration.

    Combines pre-trained PLM embeddings with VAE for drug resistance prediction.
    """

    def __init__(self, cfg: PLMConfig):
        super().__init__()
        self.cfg = cfg

        # Protein language model (mock for testing)
        self.plm = MockESM2(embed_dim=cfg.plm_dim)
        if cfg.freeze_plm:
            for param in self.plm.parameters():
                param.requires_grad = False

        # PLM-based encoder
        self.encoder = PLMVAEEncoder(cfg)

        # Decoder (for reconstruction loss)
        dec_layers = []
        dec_in = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            dec_layers.extend([nn.Linear(dec_in, h), nn.GELU()])
            dec_in = h
        dec_layers.append(nn.Linear(dec_in, cfg.adapter_dim))
        self.decoder = nn.Sequential(*dec_layers)

        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(cfg.latent_dim, 32),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(32, 1),
        )

    def get_plm_embeddings(self, tokens: torch.Tensor) -> torch.Tensor:
        """Get embeddings from PLM."""
        with torch.no_grad() if self.cfg.freeze_plm else torch.enable_grad():
            output = self.plm(tokens, repr_layers=[self.plm.num_layers])
            embeddings = output["representations"][self.plm.num_layers]
        return embeddings

    def forward(
        self,
        tokens: torch.Tensor = None,
        plm_embeddings: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            tokens: Token IDs for PLM (batch, seq_len)
            plm_embeddings: Pre-computed PLM embeddings (batch, seq_len, dim)
        """
        # Get PLM embeddings if not provided
        if plm_embeddings is None:
            if tokens is None:
                raise ValueError("Must provide either tokens or plm_embeddings")
            plm_embeddings = self.get_plm_embeddings(tokens)

        # Encode
        mu, logvar = self.encoder(plm_embeddings)

        # Reparameterize
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)

        # Decode (for reconstruction loss on adapter space)
        reconstructed = self.decoder(z)

        # Predict resistance
        prediction = self.predictor(z).squeeze(-1)

        return {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "reconstructed": reconstructed,
            "prediction": prediction,
            "plm_embeddings": plm_embeddings.mean(dim=1) if plm_embeddings.dim() == 3 else plm_embeddings,
        }


class PLMFinetuner(nn.Module):
    """Fine-tuning wrapper for PLM-based VAE.

    Implements gradual unfreezing strategy:
    1. Train adapter + VAE with frozen PLM
    2. Unfreeze last N PLM layers
    3. Fine-tune end-to-end with lower LR for PLM
    """

    def __init__(self, model: ProteinLMVAE):
        super().__init__()
        self.model = model
        self.unfrozen_layers = 0

    def unfreeze_layers(self, n_layers: int):
        """Unfreeze last n PLM layers."""
        total_layers = self.model.plm.num_layers
        layers_to_unfreeze = min(n_layers, total_layers)

        for i, layer in enumerate(reversed(list(self.model.plm.layers))):
            if i < layers_to_unfreeze:
                for param in layer.parameters():
                    param.requires_grad = True
            else:
                for param in layer.parameters():
                    param.requires_grad = False

        self.unfrozen_layers = layers_to_unfreeze
        print(f"Unfroze {layers_to_unfreeze} PLM layers")

    def get_parameter_groups(self, plm_lr: float = 1e-5, other_lr: float = 1e-3):
        """Get parameter groups with different learning rates."""
        plm_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "plm" in name:
                    plm_params.append(param)
                else:
                    other_params.append(param)

        return [
            {"params": plm_params, "lr": plm_lr},
            {"params": other_params, "lr": other_lr},
        ]

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def tokenize_sequence(sequence: str, max_len: int = 512) -> torch.Tensor:
    """Tokenize amino acid sequence for ESM-2.

    ESM-2 uses a specific alphabet:
    - 0: <cls>
    - 1: <pad>
    - 2: <eos>
    - 3: <unk>
    - 4-23: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y
    - 24-32: special tokens
    """
    aa_to_idx = {
        "A": 4,
        "C": 5,
        "D": 6,
        "E": 7,
        "F": 8,
        "G": 9,
        "H": 10,
        "I": 11,
        "K": 12,
        "L": 13,
        "M": 14,
        "N": 15,
        "P": 16,
        "Q": 17,
        "R": 18,
        "S": 19,
        "T": 20,
        "V": 21,
        "W": 22,
        "Y": 23,
    }

    # Add CLS token
    tokens = [0]  # <cls>

    for aa in sequence.upper()[: max_len - 2]:
        if aa in aa_to_idx:
            tokens.append(aa_to_idx[aa])
        else:
            tokens.append(3)  # <unk>

    # Add EOS token
    tokens.append(2)  # <eos>

    # Pad to max_len
    while len(tokens) < max_len:
        tokens.append(1)  # <pad>

    return torch.tensor(tokens, dtype=torch.long)


class HybridVAE(nn.Module):
    """Hybrid model combining PLM embeddings with one-hot encoding.

    Uses both:
    1. PLM embeddings (semantic understanding)
    2. One-hot encoding (exact mutation information)

    This provides best of both worlds for drug resistance prediction.
    """

    def __init__(self, cfg: PLMConfig, input_dim: int):
        super().__init__()
        self.cfg = cfg
        self.input_dim = input_dim

        # PLM branch
        self.plm = MockESM2(embed_dim=cfg.plm_dim)
        if cfg.freeze_plm:
            for param in self.plm.parameters():
                param.requires_grad = False

        self.plm_adapter = PLMAdapter(cfg.plm_dim, cfg.adapter_dim, cfg.adapter_dropout)

        # One-hot branch
        self.onehot_encoder = nn.Sequential(
            nn.Linear(input_dim, cfg.hidden_dims[0]),
            nn.GELU(),
            nn.LayerNorm(cfg.hidden_dims[0]),
            nn.Dropout(cfg.dropout),
        )

        # Fusion
        fusion_dim = cfg.adapter_dim + cfg.hidden_dims[0]
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, cfg.hidden_dims[1]),
            nn.GELU(),
            nn.LayerNorm(cfg.hidden_dims[1]),
            nn.Dropout(cfg.dropout),
        )

        # Latent space
        self.fc_mu = nn.Linear(cfg.hidden_dims[1], cfg.latent_dim)
        self.fc_logvar = nn.Linear(cfg.hidden_dims[1], cfg.latent_dim)

        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(cfg.latent_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x_onehot: torch.Tensor,
        tokens: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with both inputs."""

        # One-hot branch
        h_onehot = self.onehot_encoder(x_onehot)

        # PLM branch (if tokens provided)
        if tokens is not None:
            with torch.no_grad() if self.cfg.freeze_plm else torch.enable_grad():
                plm_out = self.plm(tokens, repr_layers=[self.plm.num_layers])
                plm_embed = plm_out["representations"][self.plm.num_layers].mean(dim=1)
            h_plm = self.plm_adapter(plm_embed)
        else:
            # Use zeros if no PLM input (fallback)
            h_plm = torch.zeros(x_onehot.size(0), self.cfg.adapter_dim, device=x_onehot.device)

        # Fusion
        h_fused = torch.cat([h_plm, h_onehot], dim=-1)
        h = self.fusion(h_fused)

        # Latent
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)

        # Predict
        prediction = self.predictor(z).squeeze(-1)

        return {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "prediction": prediction,
        }


if __name__ == "__main__":
    print("Testing Protein Language Model Integration")
    print("=" * 60)

    # Test PLM VAE
    cfg = PLMConfig()
    model = ProteinLMVAE(cfg)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"ProteinLMVAE parameters: {n_params:,}")

    # Test with mock tokens
    batch_size = 4
    seq_len = 100
    tokens = torch.randint(4, 24, (batch_size, seq_len))

    out = model(tokens=tokens)
    print(f"Output keys: {list(out.keys())}")
    print(f"Prediction shape: {out['prediction'].shape}")
    print(f"Latent shape: {out['z'].shape}")

    # Test tokenization
    print("\nTesting tokenization...")
    seq = "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMNLPGRWKPKMIGGIGGFIKVR"
    tokens = tokenize_sequence(seq, max_len=64)
    print(f"Sequence length: {len(seq)}")
    print(f"Token shape: {tokens.shape}")

    # Test Hybrid VAE
    print("\nTesting HybridVAE...")
    hybrid = HybridVAE(cfg, input_dim=99 * 22)

    x_onehot = torch.randn(batch_size, 99 * 22)
    tokens = torch.randint(4, 24, (batch_size, 101))

    out_hybrid = hybrid(x_onehot, tokens)
    print(f"Hybrid prediction shape: {out_hybrid['prediction'].shape}")

    # Test fine-tuner
    print("\nTesting PLMFinetuner...")
    finetuner = PLMFinetuner(model)
    finetuner.unfreeze_layers(2)

    param_groups = finetuner.get_parameter_groups()
    print(f"Parameter groups: {len(param_groups)}")
    print(f"  PLM params: {len(param_groups[0]['params'])}")
    print(f"  Other params: {len(param_groups[1]['params'])}")

    print("\n" + "=" * 60)
    print("Protein Language Model integration working!")
