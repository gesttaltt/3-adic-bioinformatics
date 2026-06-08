# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Epsilon-VAE: Meta-Learning Explorer for Checkpoint Space.

Named 'Epsilon' from 'E' for 'Explorer' - this VAE explores the manifold
of model checkpoints to discover optimal training configurations.

Capabilities:
- Metric prediction from checkpoint weights (without running the model)
- Checkpoint interpolation in latent space
- Pareto frontier discovery for coverage-structure tradeoffs
- Guided initialization for future training runs
- Future-proof: trained on past checkpoints, generalizes to unseen ones

Validation Strategy:
    Train on historical checkpoints, validate on recent/unseen ones.
    This ensures the model generalizes to future training runs.

Usage:
    # Collect checkpoint data
    dataset = collect_checkpoint_dataset("outputs/models")

    # Split temporally: train on older, validate on newer
    train_data, val_data = split_by_date(dataset, cutoff="2025-12-26")

    # Train Epsilon-VAE
    epsilon_vae = EpsilonVAE(latent_dim=32)
    train_epsilon_vae(epsilon_vae, train_data)

    # Validate on unseen checkpoints
    validate_epsilon_vae(epsilon_vae, val_data)

    # Find Pareto frontier
    pareto_z, pareto_metrics = find_pareto_frontier(epsilon_vae)

    # Interpolate between frozen and unfrozen checkpoints
    interpolated = interpolate_checkpoints(epsilon_vae, ckpt_frozen, ckpt_unfrozen)
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class WeightBlockEmbedder(nn.Module):
    """Embeds weight matrices into fixed-size vectors."""

    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        # Adaptive pooling for variable-sized weights
        self.pool = nn.AdaptiveAvgPool1d(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, weight_blocks: list[Tensor]) -> Tensor:
        """
        Embed list of weight tensors.

        Args:
            weight_blocks: List of weight tensors (various shapes)

        Returns:
            Tensor of shape [n_blocks, embed_dim]
        """
        embeddings = []
        for w in weight_blocks:
            # Flatten and pool to fixed size
            flat = w.flatten().unsqueeze(0).unsqueeze(0)  # [1, 1, N]
            pooled = self.pool(flat).squeeze(0).squeeze(0)  # [embed_dim]
            embeddings.append(pooled)

        stacked = torch.stack(embeddings, dim=0)  # [n_blocks, embed_dim]
        return self.norm(stacked)


class CheckpointEncoder(nn.Module):
    """Encodes checkpoint weights into latent space."""

    def __init__(self, embed_dim: int = 64, latent_dim: int = 32, n_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim

        self.block_embedder = WeightBlockEmbedder(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads, batch_first=True)
        self.to_latent = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, latent_dim * 2),  # mu, logvar
        )

    def forward(self, weight_blocks: list[Tensor]) -> tuple[Tensor, Tensor]:
        """
        Encode weight blocks to latent distribution.

        Args:
            weight_blocks: List of weight tensors

        Returns:
            mu, logvar: Latent distribution parameters
        """
        # Embed blocks
        blocks = self.block_embedder(weight_blocks)  # [n_blocks, embed_dim]
        blocks = blocks.unsqueeze(0)  # [1, n_blocks, embed_dim]

        # Self-attention
        attended, _ = self.attention(blocks, blocks, blocks)

        # Pool and project
        pooled = attended.mean(dim=1)  # [1, embed_dim]
        params = self.to_latent(pooled)  # [1, latent_dim * 2]

        mu, logvar = params.chunk(2, dim=-1)
        return mu.squeeze(0), logvar.squeeze(0)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick for sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class MetricPredictor(nn.Module):
    """Predicts performance metrics from latent position."""

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # [coverage, dist_corr, rad_hier]
        )

    def forward(self, z: Tensor) -> Tensor:
        """
        Predict metrics from latent.

        Args:
            z: Latent vector [latent_dim] or [batch, latent_dim]

        Returns:
            Metrics [3] or [batch, 3]: [coverage, dist_corr, rad_hier]
        """
        return self.net(z)


class CheckpointDecoder(nn.Module):
    """Decodes latent vector back to weight blocks."""

    def __init__(
        self,
        latent_dim: int = 32,
        block_sizes: list[int] | None = None,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.block_sizes = block_sizes or [1024, 1024, 1024]  # Default placeholder

        total_size = sum(self.block_sizes)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, total_size),
        )

    def forward(self, z: Tensor) -> list[Tensor]:
        """
        Decode latent to weight blocks.

        Args:
            z: Latent vector [latent_dim]

        Returns:
            List of weight tensors matching block_sizes
        """
        flat = self.net(z)

        # Split into blocks
        blocks = []
        offset = 0
        for size in self.block_sizes:
            blocks.append(flat[offset : offset + size])
            offset += size

        return blocks


class EpsilonVAE(nn.Module):
    """
    Epsilon-VAE for checkpoint exploration.

    Learns the manifold of model checkpoints and enables:
    - Metric prediction without running the model
    - Interpolation between checkpoints
    - Pareto frontier discovery
    """

    def __init__(
        self,
        embed_dim: int = 64,
        latent_dim: int = 32,
        block_sizes: list[int] | None = None,
        n_heads: int = 4,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = CheckpointEncoder(embed_dim, latent_dim, n_heads)
        self.metric_predictor = MetricPredictor(latent_dim)
        self.decoder = CheckpointDecoder(latent_dim, block_sizes)

    def encode(self, weight_blocks: list[Tensor]) -> tuple[Tensor, Tensor]:
        """Encode weights to latent distribution."""
        return self.encoder(weight_blocks)

    def decode(self, z: Tensor) -> list[Tensor]:
        """Decode latent to weight blocks."""
        return self.decoder(z)

    def predict_metrics(self, z: Tensor) -> Tensor:
        """Predict metrics from latent."""
        return self.metric_predictor(z)

    def forward(self, weight_blocks: list[Tensor]) -> tuple[Tensor, Tensor, Tensor, list[Tensor]]:
        """
        Full forward pass.

        Returns:
            mu, logvar, metrics_pred, weights_recon
        """
        mu, logvar = self.encode(weight_blocks)
        z = self.encoder.reparameterize(mu, logvar)
        metrics_pred = self.predict_metrics(z)
        weights_recon = self.decode(z)

        return mu, logvar, metrics_pred, weights_recon


def epsilon_vae_loss(
    metrics_pred: Tensor,
    metrics_true: Tensor,
    mu: Tensor,
    logvar: Tensor,
    weights_pred: list[Tensor] | None = None,
    weights_true: list[Tensor] | None = None,
    beta: float = 0.1,
    recon_weight: float = 0.1,
) -> dict[str, Tensor]:
    """
    Compute Epsilon-VAE loss.

    Args:
        metrics_pred: Predicted [coverage, dist_corr, rad_hier]
        metrics_true: True metrics
        mu, logvar: Latent distribution parameters
        weights_pred, weights_true: Optional weight reconstruction
        beta: KL weight
        recon_weight: Weight reconstruction weight

    Returns:
        Dict with loss components
    """
    # Metric prediction loss (primary)
    metric_loss = F.mse_loss(metrics_pred, metrics_true)

    # KL divergence
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total = metric_loss + beta * kl_loss

    losses = {
        "metric_loss": metric_loss,
        "kl_loss": kl_loss,
        "total": total,
    }

    # Optional weight reconstruction
    if weights_pred is not None and weights_true is not None:
        recon_loss = sum(F.mse_loss(wp, wt) for wp, wt in zip(weights_pred, weights_true, strict=False)) / len(
            weights_pred
        )
        losses["recon_loss"] = recon_loss
        losses["total"] = total + recon_weight * recon_loss

    return losses


def extract_key_weights(state_dict: dict, key_patterns: list[str] | None = None) -> list[Tensor]:
    """
    Extract key weight matrices from state dict.

    Args:
        state_dict: Model state dict
        key_patterns: Patterns to match (default: projection layers)

    Returns:
        List of weight tensors
    """
    if key_patterns is None:
        key_patterns = [
            "projection",
            "encoder",
            "fc",
            "linear",
        ]

    weights = []
    for name, param in state_dict.items():
        if any(pattern in name.lower() for pattern in key_patterns):
            if "weight" in name.lower() and param.dim() >= 2:
                weights.append(param.detach().clone())

    return weights


def collect_checkpoint_dataset(checkpoint_dir: str | Path, max_checkpoints: int = 500) -> list[dict]:
    """
    Collect (weights, metrics) pairs from checkpoint directory.

    Args:
        checkpoint_dir: Path to checkpoint directory
        max_checkpoints: Maximum checkpoints to load

    Returns:
        List of {weights, coverage, dist_corr, rad_hier} dicts
    """
    checkpoint_dir = Path(checkpoint_dir)
    dataset = []

    for run_dir in checkpoint_dir.iterdir():
        if not run_dir.is_dir():
            continue

        for ckpt_path in run_dir.glob("*.pt"):
            if len(dataset) >= max_checkpoints:
                break

            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")

                # Extract weights
                state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", {}))
                weights = extract_key_weights(state_dict)

                if not weights:
                    continue

                # Extract metrics
                metrics = ckpt.get("metrics", {})

                dataset.append(
                    {
                        "weights": weights,
                        "coverage": metrics.get("coverage", 0.0),
                        "dist_corr": metrics.get("distance_corr_A", 0.0),
                        "rad_hier": metrics.get("radial_corr_A", 0.0),
                        "path": str(ckpt_path),
                    }
                )

            except Exception as e:
                print(f"Error loading {ckpt_path}: {e}")
                continue

    return dataset


def is_pareto_efficient(costs: Tensor) -> Tensor:
    """
    Find Pareto-efficient points.

    Args:
        costs: [n_points, n_objectives] - lower is better

    Returns:
        Boolean mask of Pareto-efficient points
    """
    n_points = costs.shape[0]
    is_efficient = torch.ones(n_points, dtype=torch.bool)

    for i in range(n_points):
        # Check if any other point strictly dominates point i
        # Point j dominates i if: all(costs[j] <= costs[i]) AND any(costs[j] < costs[i])
        strictly_dominates = (costs <= costs[i]).all(dim=1) & (costs < costs[i]).any(dim=1)
        strictly_dominates[i] = False  # Don't compare with self

        if strictly_dominates.any():
            is_efficient[i] = False

    return is_efficient


def find_pareto_frontier(epsilon_vae: EpsilonVAE, n_samples: int = 1000, device: str = "cpu") -> tuple[Tensor, Tensor]:
    """
    Sample latent space to find Pareto-optimal configurations.

    Optimizes for: high coverage, high dist_corr (both maximized)

    Args:
        epsilon_vae: Trained Epsilon-VAE
        n_samples: Number of latent samples
        device: Device to run on

    Returns:
        pareto_z: Latent vectors on Pareto frontier
        pareto_metrics: Corresponding metrics
    """
    epsilon_vae.eval()
    epsilon_vae.to(device)

    with torch.no_grad():
        z_samples = torch.randn(n_samples, epsilon_vae.latent_dim, device=device)
        metrics = epsilon_vae.predict_metrics(z_samples)

        # Convert to costs (negate to minimize)
        # We want to maximize coverage and dist_corr
        costs = -metrics[:, :2]  # [coverage, dist_corr]

        pareto_mask = is_pareto_efficient(costs)

        return z_samples[pareto_mask], metrics[pareto_mask]


def interpolate_checkpoints(
    epsilon_vae: EpsilonVAE,
    weights_a: list[Tensor],
    weights_b: list[Tensor],
    n_steps: int = 10,
) -> list[dict]:
    """
    Interpolate between two checkpoints in latent space.

    Args:
        epsilon_vae: Trained Epsilon-VAE
        weights_a, weights_b: Weight blocks from two checkpoints
        n_steps: Number of interpolation steps

    Returns:
        List of {alpha, z, predicted_metrics, weights} dicts
    """
    epsilon_vae.eval()

    with torch.no_grad():
        mu_a, _ = epsilon_vae.encode(weights_a)
        mu_b, _ = epsilon_vae.encode(weights_b)

        interpolated = []
        for alpha in torch.linspace(0, 1, n_steps):
            z_interp = (1 - alpha) * mu_a + alpha * mu_b
            metrics = epsilon_vae.predict_metrics(z_interp)
            weights = epsilon_vae.decode(z_interp)

            interpolated.append(
                {
                    "alpha": alpha.item(),
                    "z": z_interp,
                    "predicted_metrics": {
                        "coverage": metrics[0].item(),
                        "dist_corr": metrics[1].item(),
                        "rad_hier": metrics[2].item(),
                    },
                    "weights": weights,
                }
            )

        return interpolated
