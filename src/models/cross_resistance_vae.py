"""Cross-Resistance VAE for NRTI Drugs.

NRTI drugs have complex cross-resistance patterns:
- TAMs (41, 67, 70, 210, 215, 219) affect AZT, D4T, ABC, TDF, DDI
- M184V (184) primarily affects 3TC, FTC, ABC but resensitizes to AZT
- K65R (65) affects TDF, ABC, DDI, 3TC but antagonizes TAMs

This module models these interactions using:
1. Drug relationship graphs
2. Multi-output prediction with shared latent space
3. Cross-drug attention mechanisms

References:
- Johnson et al., 2008: TAM pathways
- Whitcomb et al., 2003: M184V resensitization
- Parikh et al., 2006: K65R/TAM antagonism
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Cross-Resistance Knowledge Base
# =============================================================================

# Cross-resistance matrix: positive = cross-resistant, negative = antagonism
# Values represent expected correlation in resistance patterns
CROSS_RESISTANCE_MATRIX = {
    #      AZT    D4T    ABC    TDF    DDI    3TC
    "AZT": {"AZT": 1.00, "D4T": 0.85, "ABC": 0.40, "TDF": 0.30, "DDI": 0.35, "3TC": -0.15},
    "D4T": {"AZT": 0.85, "D4T": 1.00, "ABC": 0.45, "TDF": 0.35, "DDI": 0.40, "3TC": -0.10},
    "ABC": {"AZT": 0.40, "D4T": 0.45, "ABC": 1.00, "TDF": 0.55, "DDI": 0.50, "3TC": 0.35},
    "TDF": {"AZT": 0.30, "D4T": 0.35, "TDF": 1.00, "ABC": 0.55, "DDI": 0.45, "3TC": 0.25},
    "DDI": {"AZT": 0.35, "D4T": 0.40, "ABC": 0.50, "TDF": 0.45, "DDI": 1.00, "3TC": 0.20},
    "3TC": {"AZT": -0.15, "D4T": -0.10, "ABC": 0.35, "TDF": 0.25, "DDI": 0.20, "3TC": 1.00},
}

# Mutation effects on each drug (positive = increases resistance)
MUTATION_DRUG_EFFECTS = {
    # TAM-1 pathway
    "M41L": {"AZT": 0.8, "D4T": 0.7, "ABC": 0.3, "TDF": 0.2, "DDI": 0.2, "3TC": 0.0},
    "L210W": {"AZT": 0.6, "D4T": 0.5, "ABC": 0.2, "TDF": 0.1, "DDI": 0.1, "3TC": 0.0},
    "T215Y": {"AZT": 0.9, "D4T": 0.8, "ABC": 0.4, "TDF": 0.3, "DDI": 0.3, "3TC": 0.0},
    # TAM-2 pathway
    "D67N": {"AZT": 0.5, "D4T": 0.4, "ABC": 0.2, "TDF": 0.1, "DDI": 0.1, "3TC": 0.0},
    "K70R": {"AZT": 0.6, "D4T": 0.5, "ABC": 0.2, "TDF": 0.3, "DDI": 0.2, "3TC": 0.0},
    "T215F": {"AZT": 0.7, "D4T": 0.6, "ABC": 0.3, "TDF": 0.2, "DDI": 0.2, "3TC": 0.0},
    "K219Q": {"AZT": 0.4, "D4T": 0.3, "ABC": 0.1, "TDF": 0.1, "DDI": 0.1, "3TC": 0.0},
    # M184V - resensitizes to AZT!
    "M184V": {"AZT": -0.2, "D4T": 0.0, "ABC": 0.6, "TDF": 0.1, "DDI": 0.1, "3TC": 1.0},
    # K65R - antagonizes TAMs
    "K65R": {"AZT": -0.3, "D4T": -0.2, "ABC": 0.7, "TDF": 0.9, "DDI": 0.5, "3TC": 0.4},
    # L74V
    "L74V": {"AZT": 0.0, "D4T": 0.0, "ABC": 0.6, "TDF": 0.1, "DDI": 0.7, "3TC": 0.0},
}


@dataclass
class CrossResistanceConfig:
    """Configuration for cross-resistance VAE."""

    input_dim: int
    latent_dim: int = 32
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    drug_names: list[str] = field(default_factory=lambda: ["AZT", "D4T", "ABC", "TDF", "DDI", "3TC"])
    n_positions: int = 240
    dropout: float = 0.1
    use_cross_attention: bool = True
    use_mutation_embeddings: bool = True
    ranking_weight: float = 0.3


class MutationEmbedding(nn.Module):
    """Embed known mutation effects into drug predictions."""

    def __init__(self, n_positions: int, n_drugs: int, embedding_dim: int = 32):
        super().__init__()
        self.n_positions = n_positions
        self.n_drugs = n_drugs

        # Position embeddings for key mutation sites
        key_positions = [41, 62, 65, 67, 69, 70, 74, 75, 77, 115, 116, 151, 184, 210, 215, 219]
        self.key_positions = key_positions

        # Learnable embeddings for each key position
        self.position_embed = nn.Embedding(len(key_positions), embedding_dim)

        # Project to drug-specific effects
        self.drug_projection = nn.Linear(embedding_dim, n_drugs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute mutation-based drug effect embeddings.

        Args:
            x: One-hot encoded sequences (batch, n_positions * 22)

        Returns:
            Drug effect embeddings (batch, n_drugs)
        """
        batch_size = x.size(0)
        device = x.device

        # Reshape to (batch, n_positions, 22)
        x = x.view(batch_size, self.n_positions, 22)

        # Check for mutations at key positions
        effects = []
        for i, pos in enumerate(self.key_positions):
            if pos <= self.n_positions:
                pos_embed = self.position_embed(torch.tensor([i], device=device))
                # Check if position is mutated (not reference AA)
                # Simple: sum of non-gap amino acids
                pos_data = x[:, pos - 1, :]  # 0-indexed
                is_mutated = (pos_data[:, :-1].sum(dim=-1) > 0).float().unsqueeze(1)  # Exclude gap
                effects.append(is_mutated * pos_embed)

        if effects:
            combined = torch.stack(effects, dim=1).sum(dim=1)  # (batch, embedding_dim)
            drug_effects = self.drug_projection(combined)  # (batch, n_drugs)
        else:
            drug_effects = torch.zeros(batch_size, len(self.key_positions), device=device)

        return drug_effects


class CrossDrugAttention(nn.Module):
    """Attention mechanism for cross-drug information sharing."""

    def __init__(self, n_drugs: int, latent_dim: int, n_heads: int = 4):
        super().__init__()
        self.n_drugs = n_drugs
        self.latent_dim = latent_dim

        # Drug embeddings
        self.drug_embed = nn.Embedding(n_drugs, latent_dim)

        # Cross-drug attention
        self.attention = nn.MultiheadAttention(latent_dim, n_heads, batch_first=True)

        # Output projection
        self.output_proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Apply cross-drug attention to latent representation.

        Args:
            z: Latent representation (batch, latent_dim)

        Returns:
            Drug-contextualized representations (batch, n_drugs, latent_dim)
        """
        batch_size = z.size(0)
        device = z.device

        # Get drug embeddings
        drug_indices = torch.arange(self.n_drugs, device=device)
        drug_embeds = self.drug_embed(drug_indices)  # (n_drugs, latent_dim)

        # Expand for batch
        drug_embeds = drug_embeds.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, n_drugs, latent_dim)

        # Add latent to each drug embedding
        z_expanded = z.unsqueeze(1).expand(-1, self.n_drugs, -1)  # (batch, n_drugs, latent_dim)
        query = drug_embeds + z_expanded

        # Self-attention over drugs
        attn_out, _ = self.attention(query, query, query)

        # Project
        output = self.output_proj(attn_out)  # (batch, n_drugs, latent_dim)

        return output


class CrossResistanceVAE(nn.Module):
    """VAE with cross-resistance modeling for NRTI drugs."""

    def __init__(self, cfg: CrossResistanceConfig):
        super().__init__()
        self.cfg = cfg
        self.drug_names = cfg.drug_names
        self.n_drugs = len(cfg.drug_names)

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

        # Latent
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

        # Cross-drug attention
        if cfg.use_cross_attention:
            self.cross_attention = CrossDrugAttention(self.n_drugs, cfg.latent_dim)
        else:
            self.cross_attention = None

        # Mutation embeddings
        if cfg.use_mutation_embeddings:
            self.mutation_embed = MutationEmbedding(cfg.n_positions, self.n_drugs)
        else:
            self.mutation_embed = None

        # Drug-specific prediction heads
        self.drug_heads = nn.ModuleDict(
            {
                drug: nn.Sequential(
                    nn.Linear(cfg.latent_dim + (self.n_drugs if cfg.use_mutation_embeddings else 0), 32),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(32, 1),
                )
                for drug in cfg.drug_names
            }
        )

        # Cross-resistance regularization weight
        self.register_buffer("cross_resistance_matrix", self._build_cross_resistance_tensor())

    def _build_cross_resistance_tensor(self) -> torch.Tensor:
        """Build cross-resistance matrix as tensor."""
        n = len(self.drug_names)
        matrix = torch.zeros(n, n)
        for i, drug1 in enumerate(self.drug_names):
            for j, drug2 in enumerate(self.drug_names):
                if drug1 in CROSS_RESISTANCE_MATRIX and drug2 in CROSS_RESISTANCE_MATRIX[drug1]:
                    matrix[i, j] = CROSS_RESISTANCE_MATRIX[drug1][drug2]
        return matrix

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        x: torch.Tensor,
        target_drugs: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with optional drug selection.

        Args:
            x: Input sequences (batch, input_dim)
            target_drugs: Drugs to predict (None = all)

        Returns:
            Dict with predictions, latent, reconstruction
        """
        x.size(0)

        # Encode
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)

        # Decode
        x_recon = self.decoder(z)

        # Cross-drug attention
        if self.cross_attention is not None:
            z_drugs = self.cross_attention(z)  # (batch, n_drugs, latent_dim)
        else:
            z_drugs = z.unsqueeze(1).expand(-1, self.n_drugs, -1)

        # Mutation embeddings
        if self.mutation_embed is not None:
            mut_effects = self.mutation_embed(x)  # (batch, n_drugs)
        else:
            mut_effects = None

        # Predict for each drug
        predictions = {}
        drugs_to_predict = target_drugs if target_drugs else self.drug_names

        for i, drug in enumerate(self.drug_names):
            if drug not in drugs_to_predict:
                continue

            z_drug = z_drugs[:, i, :]  # (batch, latent_dim)

            if mut_effects is not None:
                z_drug = torch.cat([z_drug, mut_effects], dim=-1)

            pred = self.drug_heads[drug](z_drug).squeeze(-1)
            predictions[drug] = pred

        result = {
            "x_recon": x_recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "predictions": predictions,
        }

        # Add single prediction if only one drug
        if len(predictions) == 1:
            result["prediction"] = list(predictions.values())[0]

        return result

    def compute_cross_resistance_loss(
        self,
        predictions: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute cross-resistance consistency loss.

        Encourages predictions to follow known cross-resistance patterns.
        """
        if len(predictions) < 2:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        # Stack predictions
        drug_preds = []
        drug_indices = []
        for i, drug in enumerate(self.drug_names):
            if drug in predictions:
                drug_preds.append(predictions[drug])
                drug_indices.append(i)

        if len(drug_preds) < 2:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        pred_stack = torch.stack(drug_preds, dim=1)  # (batch, n_predicted_drugs)

        # Compute pairwise correlations
        pred_centered = pred_stack - pred_stack.mean(dim=0, keepdim=True)
        pred_std = pred_stack.std(dim=0, keepdim=True) + 1e-8

        # Correlation matrix of predictions
        n_pred = len(drug_indices)
        pred_corr = torch.zeros(n_pred, n_pred, device=pred_stack.device)
        for i in range(n_pred):
            for j in range(n_pred):
                if i != j:
                    corr = (pred_centered[:, i] * pred_centered[:, j]).mean()
                    corr = corr / (pred_std[:, i] * pred_std[:, j]).mean()
                    pred_corr[i, j] = corr

        # Expected correlation from cross-resistance matrix
        expected_corr = self.cross_resistance_matrix[drug_indices][:, drug_indices]

        # Loss: MSE between predicted and expected correlations
        loss = F.mse_loss(pred_corr, expected_corr)

        return loss


def compute_cross_resistance_loss(
    cfg: CrossResistanceConfig,
    out: dict[str, torch.Tensor],
    x: torch.Tensor,
    targets: dict[str, torch.Tensor],
    cross_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Compute full loss with cross-resistance regularization.

    Args:
        cfg: Configuration
        out: Model output
        x: Input
        targets: Target values per drug
        cross_weight: Weight for cross-resistance loss

    Returns:
        Dict of losses
    """
    losses = {}

    # Reconstruction
    losses["recon"] = F.mse_loss(out["x_recon"], x)

    # KL divergence
    kl = -0.5 * torch.sum(1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp())
    losses["kl"] = 0.001 * kl / x.size(0)

    # Drug-specific ranking losses
    predictions = out["predictions"]
    drug_losses = []

    for drug, target in targets.items():
        if drug not in predictions:
            continue

        pred = predictions[drug]

        # Ranking loss (correlation)
        p_c = pred - pred.mean()
        t_c = target - target.mean()
        p_std = torch.sqrt(torch.sum(p_c**2) + 1e-8)
        t_std = torch.sqrt(torch.sum(t_c**2) + 1e-8)
        corr = torch.sum(p_c * t_c) / (p_std * t_std)

        drug_loss = cfg.ranking_weight * (-corr)
        losses[f"{drug}_loss"] = drug_loss
        drug_losses.append(drug_loss)

    if drug_losses:
        losses["drug_total"] = sum(drug_losses) / len(drug_losses)

    # Cross-resistance consistency loss
    if len(predictions) >= 2:
        # Compute pairwise prediction correlations
        drug_names = list(predictions.keys())
        pred_stack = torch.stack([predictions[d] for d in drug_names], dim=1)

        # Center predictions
        pred_centered = pred_stack - pred_stack.mean(dim=0, keepdim=True)

        # Compute correlation matrix
        n_drugs = len(drug_names)
        cross_loss = 0.0
        count = 0

        for i in range(n_drugs):
            for j in range(i + 1, n_drugs):
                # Predicted correlation
                pred_corr = (pred_centered[:, i] * pred_centered[:, j]).mean()
                pred_corr = pred_corr / (pred_centered[:, i].std() * pred_centered[:, j].std() + 1e-8)

                # Expected correlation
                drug_i, drug_j = drug_names[i], drug_names[j]
                if drug_i in CROSS_RESISTANCE_MATRIX and drug_j in CROSS_RESISTANCE_MATRIX[drug_i]:
                    expected = CROSS_RESISTANCE_MATRIX[drug_i][drug_j]

                    # Loss: deviation from expected correlation
                    cross_loss = cross_loss + (pred_corr - expected) ** 2
                    count += 1

        if count > 0:
            losses["cross_resistance"] = cross_weight * cross_loss / count

    losses["total"] = losses["recon"] + losses["kl"] + losses.get("drug_total", 0) + losses.get("cross_resistance", 0)

    return losses


if __name__ == "__main__":
    print("Testing Cross-Resistance VAE")
    print("=" * 60)

    # Create config
    cfg = CrossResistanceConfig(
        input_dim=240 * 22,
        n_positions=240,
        drug_names=["AZT", "D4T", "ABC", "TDF", "DDI", "3TC"],
    )

    # Create model
    model = CrossResistanceVAE(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Test forward pass
    x = torch.randn(8, 240 * 22)
    out = model(x)

    print(f"\nOutput keys: {list(out.keys())}")
    print(f"Predictions: {list(out['predictions'].keys())}")
    print(f"Latent shape: {out['z'].shape}")

    # Test loss computation
    targets = {
        "AZT": torch.rand(8),
        "3TC": torch.rand(8),
        "TDF": torch.rand(8),
    }

    losses = compute_cross_resistance_loss(cfg, out, x, targets)
    print(f"\nLosses: {list(losses.keys())}")
    for name, value in losses.items():
        print(f"  {name}: {value.item():.4f}")

    print("\n" + "=" * 60)
    print("Cross-Resistance VAE working!")
