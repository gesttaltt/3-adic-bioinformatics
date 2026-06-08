"""Multi-task VAE for simultaneous drug resistance prediction.

Instead of training separate models for each drug, this module trains
a shared encoder with drug-specific prediction heads. Benefits:

1. Shared representation captures common resistance patterns
2. Transfer learning to new/rare drugs
3. Regularization through multi-task learning
4. Better sample efficiency

References:
- Caruana (1997): Multitask Learning
- Ruder (2017): An Overview of Multi-Task Learning
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MultiTaskConfig:
    """Configuration for multi-task VAE."""

    input_dim: int
    latent_dim: int = 16
    hidden_dims: list[int] = field(default_factory=lambda: [128, 64, 32])
    drug_names: list[str] = field(default_factory=list)
    dropout: float = 0.1
    use_ranking_loss: bool = True
    ranking_weight: float = 0.3
    share_prediction_layers: bool = False  # Share some layers between drugs


class SharedEncoder(nn.Module):
    """Shared encoder for all drugs."""

    def __init__(self, cfg: MultiTaskConfig):
        super().__init__()

        layers = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, h),
                    nn.GELU(),
                    nn.BatchNorm1d(h),
                    nn.Dropout(cfg.dropout),
                ]
            )
            in_dim = h

        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(in_dim, cfg.latent_dim)
        self.fc_logvar = nn.Linear(in_dim, cfg.latent_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return {"mu": mu, "logvar": logvar, "h": h}


class SharedDecoder(nn.Module):
    """Shared decoder for reconstruction."""

    def __init__(self, cfg: MultiTaskConfig):
        super().__init__()

        layers = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            layers.extend(
                [
                    nn.Linear(in_dim, h),
                    nn.GELU(),
                    nn.BatchNorm1d(h),
                    nn.Dropout(cfg.dropout),
                ]
            )
            in_dim = h

        layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class DrugHead(nn.Module):
    """Drug-specific prediction head."""

    def __init__(self, latent_dim: int, hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


class MultiTaskVAE(nn.Module):
    """VAE with shared encoder and drug-specific heads."""

    def __init__(self, cfg: MultiTaskConfig):
        super().__init__()
        self.cfg = cfg
        self.drug_names = cfg.drug_names

        # Shared components
        self.encoder = SharedEncoder(cfg)
        self.decoder = SharedDecoder(cfg)

        # Drug-specific heads
        self.drug_heads = nn.ModuleDict(
            {drug: DrugHead(cfg.latent_dim, dropout=cfg.dropout) for drug in cfg.drug_names}
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        x: torch.Tensor,
        drug: str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input sequences
            drug: Specific drug to predict (None for all)

        Returns:
            Dict with reconstruction, latent, and predictions
        """
        # Encode
        enc_out = self.encoder(x)
        z = self.reparameterize(enc_out["mu"], enc_out["logvar"])

        # Decode
        x_recon = self.decoder(z)

        result = {
            "x_recon": x_recon,
            "mu": enc_out["mu"],
            "logvar": enc_out["logvar"],
            "z": z,
        }

        # Predict
        if drug is not None:
            result["prediction"] = self.drug_heads[drug](z)
        else:
            result["predictions"] = {name: self.drug_heads[name](z) for name in self.drug_names}

        return result

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation."""
        enc_out = self.encoder(x)
        return enc_out["mu"]

    def predict_drug(self, x: torch.Tensor, drug: str) -> torch.Tensor:
        """Predict resistance for specific drug."""
        z = self.encode(x)
        return self.drug_heads[drug](z)

    def predict_all(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict resistance for all drugs."""
        z = self.encode(x)
        return {name: self.drug_heads[name](z) for name in self.drug_names}


def compute_multi_task_loss(
    cfg: MultiTaskConfig,
    out: dict[str, torch.Tensor],
    x: torch.Tensor,
    targets: dict[str, torch.Tensor],
    drug_weights: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute multi-task loss.

    Args:
        cfg: Configuration
        out: Model output
        x: Input
        targets: Dict mapping drug name to target values
        drug_weights: Optional weights for each drug loss

    Returns:
        Dict of losses
    """
    losses = {}

    # Reconstruction loss
    losses["recon"] = F.mse_loss(out["x_recon"], x)

    # KL divergence
    kl = -0.5 * torch.sum(1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp())
    losses["kl"] = 0.001 * kl / x.size(0)

    # Drug-specific losses
    if drug_weights is None:
        drug_weights = {drug: 1.0 for drug in targets}

    predictions = out.get("predictions", {})
    if "prediction" in out and len(targets) == 1:
        drug = list(targets.keys())[0]
        predictions = {drug: out["prediction"]}

    drug_losses = []
    for drug, target in targets.items():
        if drug not in predictions:
            continue

        pred = predictions[drug]
        weight = drug_weights.get(drug, 1.0)

        # MSE loss
        mse = F.mse_loss(pred, target)

        # Ranking loss
        if cfg.use_ranking_loss:
            p_c = pred - pred.mean()
            t_c = target - target.mean()
            p_std = torch.sqrt(torch.sum(p_c**2) + 1e-8)
            t_std = torch.sqrt(torch.sum(t_c**2) + 1e-8)
            corr = torch.sum(p_c * t_c) / (p_std * t_std)
            rank_loss = cfg.ranking_weight * (-corr)
            drug_loss = mse + rank_loss
        else:
            drug_loss = mse

        losses[f"{drug}_loss"] = weight * drug_loss
        drug_losses.append(weight * drug_loss)

    if drug_losses:
        losses["drug_total"] = sum(drug_losses)

    losses["total"] = losses["recon"] + losses["kl"] + losses.get("drug_total", 0)

    return losses


class GradientBalancedMultiTaskVAE(MultiTaskVAE):
    """Multi-task VAE with gradient balancing.

    Uses GradNorm-style gradient balancing to prevent any single
    drug from dominating the shared representation.
    """

    def __init__(self, cfg: MultiTaskConfig, alpha: float = 1.5):
        super().__init__(cfg)
        self.alpha = alpha

        # Learnable task weights
        self.task_weights = nn.Parameter(torch.ones(len(cfg.drug_names)))
        self.initial_losses: dict[str, float] | None = None

    def get_balanced_weights(self, current_losses: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute balanced weights based on relative loss progress."""
        if self.initial_losses is None:
            self.initial_losses = {k: v.item() for k, v in current_losses.items()}
            return {k: torch.tensor(1.0) for k in current_losses}

        # Compute relative inverse training rates
        weights = {}
        total = 0.0
        for _i, drug in enumerate(self.drug_names):
            key = f"{drug}_loss"
            if key in current_losses and key in self.initial_losses:
                ratio = current_losses[key].item() / (self.initial_losses[key] + 1e-8)
                weight = ratio**self.alpha
                weights[drug] = weight
                total += weight

        # Normalize
        if total > 0:
            weights = {k: v / total * len(weights) for k, v in weights.items()}

        return weights


class CrossDrugTransferVAE(MultiTaskVAE):
    """Multi-task VAE with explicit cross-drug transfer.

    Adds cross-drug attention to allow information flow between
    drugs that share resistance mechanisms.
    """

    def __init__(self, cfg: MultiTaskConfig, cross_drug_dim: int = 32):
        super().__init__(cfg)

        self.n_drugs = len(cfg.drug_names)
        self.cross_drug_dim = cross_drug_dim

        # Drug embeddings
        self.drug_embeddings = nn.Embedding(self.n_drugs, cross_drug_dim)

        # Cross-drug attention
        self.cross_attention = nn.MultiheadAttention(cross_drug_dim, num_heads=4, batch_first=True)

        # Modified heads that incorporate cross-drug info
        self.cross_heads = nn.ModuleDict(
            {
                drug: nn.Sequential(
                    nn.Linear(cfg.latent_dim + cross_drug_dim, 32),
                    nn.GELU(),
                    nn.Linear(32, 1),
                )
                for drug in cfg.drug_names
            }
        )

    def forward(
        self,
        x: torch.Tensor,
        drug: str | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = x.size(0)

        # Encode
        enc_out = self.encoder(x)
        z = self.reparameterize(enc_out["mu"], enc_out["logvar"])
        x_recon = self.decoder(z)

        # Drug embeddings
        drug_indices = torch.arange(self.n_drugs, device=x.device)
        drug_embeds = self.drug_embeddings(drug_indices)  # (n_drugs, cross_drug_dim)

        # Cross-drug attention
        drug_embeds = drug_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        cross_info, _ = self.cross_attention(drug_embeds, drug_embeds, drug_embeds)

        result = {
            "x_recon": x_recon,
            "mu": enc_out["mu"],
            "logvar": enc_out["logvar"],
            "z": z,
        }

        # Predictions with cross-drug info
        if drug is not None:
            drug_idx = self.drug_names.index(drug)
            combined = torch.cat([z, cross_info[:, drug_idx]], dim=-1)
            result["prediction"] = self.cross_heads[drug](combined).squeeze(-1)
        else:
            predictions = {}
            for i, name in enumerate(self.drug_names):
                combined = torch.cat([z, cross_info[:, i]], dim=-1)
                predictions[name] = self.cross_heads[name](combined).squeeze(-1)
            result["predictions"] = predictions

        return result


def create_multi_task_vae(
    input_dim: int,
    drug_names: list[str],
    variant: str = "standard",
) -> nn.Module:
    """Factory function to create multi-task VAE.

    Args:
        input_dim: Input dimension
        drug_names: List of drug names
        variant: 'standard', 'balanced', or 'cross_drug'

    Returns:
        Multi-task VAE model
    """
    cfg = MultiTaskConfig(input_dim=input_dim, drug_names=drug_names)

    if variant == "standard":
        return MultiTaskVAE(cfg)
    elif variant == "balanced":
        return GradientBalancedMultiTaskVAE(cfg)
    elif variant == "cross_drug":
        return CrossDrugTransferVAE(cfg)
    else:
        raise ValueError(f"Unknown variant: {variant}")


if __name__ == "__main__":
    print("Testing Multi-Task VAE")
    print("=" * 60)

    drugs = ["LPV", "DRV", "ATV", "IDV", "NFV", "SQV", "TPV", "FPV"]
    cfg = MultiTaskConfig(input_dim=99 * 22, drug_names=drugs)

    # Test standard
    model = MultiTaskVAE(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Standard Multi-Task VAE: {n_params:,} parameters")

    x = torch.randn(8, 99 * 22)
    out = model(x)
    print(f"Predictions for {len(out['predictions'])} drugs")

    # Test single drug prediction
    out_lpv = model(x, drug="LPV")
    print(f"LPV prediction shape: {out_lpv['prediction'].shape}")

    # Test balanced variant
    model_balanced = GradientBalancedMultiTaskVAE(cfg)
    print(f"\nBalanced variant: task weights = {model_balanced.task_weights.data}")

    # Test cross-drug variant
    model_cross = CrossDrugTransferVAE(cfg)
    out_cross = model_cross(x)
    print(f"\nCross-drug variant predictions: {len(out_cross['predictions'])}")

    print("\n" + "=" * 60)
    print("All multi-task variants working!")
