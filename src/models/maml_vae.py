"""MAML-style meta-learning for few-shot drug resistance prediction.

For new drugs with limited data (e.g., CAB, BIC), standard training fails.
Meta-learning enables rapid adaptation from few examples by:

1. Learning a good initialization across tasks (drugs)
2. Enabling quick adaptation with just 5-10 gradient steps
3. Transferring resistance patterns from related drugs

References:
- Finn et al. (2017): Model-Agnostic Meta-Learning for Fast Adaptation
- Raghu et al. (2019): Rapid Learning or Feature Reuse?
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MAMLConfig:
    """Configuration for MAML training."""

    input_dim: int
    latent_dim: int = 16
    hidden_dims: list[int] = field(default_factory=lambda: [128, 64, 32])
    inner_lr: float = 0.01  # Learning rate for task adaptation
    outer_lr: float = 0.001  # Learning rate for meta-update
    inner_steps: int = 5  # Gradient steps for adaptation
    meta_batch_size: int = 4  # Number of tasks per meta-batch
    first_order: bool = True  # Use first-order approximation (faster)
    use_ranking_loss: bool = True
    ranking_weight: float = 0.3


class MAMLEncoder(nn.Module):
    """Encoder for MAML VAE."""

    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int):
        super().__init__()

        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, h),
                    nn.GELU(),
                    nn.LayerNorm(h),  # LayerNorm instead of BatchNorm for MAML
                ]
            )
            in_dim = h

        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)


class MAMLDecoder(nn.Module):
    """Decoder for MAML VAE."""

    def __init__(self, latent_dim: int, hidden_dims: list[int], output_dim: int):
        super().__init__()

        layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            layers.extend([nn.Linear(in_dim, h), nn.GELU(), nn.LayerNorm(h)])
            in_dim = h

        layers.append(nn.Linear(in_dim, output_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class MAMLVAE(nn.Module):
    """VAE designed for MAML meta-learning."""

    def __init__(self, cfg: MAMLConfig):
        super().__init__()
        self.cfg = cfg

        self.encoder = MAMLEncoder(cfg.input_dim, cfg.hidden_dims, cfg.latent_dim)
        self.decoder = MAMLDecoder(cfg.latent_dim, cfg.hidden_dims, cfg.input_dim)
        self.predictor = nn.Linear(cfg.latent_dim, 1)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        prediction = self.predictor(z).squeeze(-1)

        return {
            "x_recon": x_recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "prediction": prediction,
        }


def compute_task_loss(
    cfg: MAMLConfig,
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Compute loss for a single task."""
    out = model(x)

    # Reconstruction
    recon = F.mse_loss(out["x_recon"], x)

    # KL
    kl = -0.5 * torch.mean(1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp())

    # Prediction
    mse = F.mse_loss(out["prediction"], y)

    # Ranking
    if cfg.use_ranking_loss:
        p_c = out["prediction"] - out["prediction"].mean()
        y_c = y - y.mean()
        p_std = torch.sqrt(torch.sum(p_c**2) + 1e-8)
        y_std = torch.sqrt(torch.sum(y_c**2) + 1e-8)
        corr = torch.sum(p_c * y_c) / (p_std * y_std)
        rank_loss = cfg.ranking_weight * (-corr)
    else:
        rank_loss = 0

    return recon + 0.001 * kl + mse + rank_loss


def clone_model(model: nn.Module) -> nn.Module:
    """Create a deep copy of model with same architecture."""
    return copy.deepcopy(model)


def adapt_model(
    model: nn.Module,
    cfg: MAMLConfig,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
) -> nn.Module:
    """Adapt model to a task using support set.

    Args:
        model: Base model
        cfg: Configuration
        support_x: Support set inputs
        support_y: Support set targets

    Returns:
        Adapted model
    """
    adapted = clone_model(model)

    for _ in range(cfg.inner_steps):
        loss = compute_task_loss(cfg, adapted, support_x, support_y)

        # Compute gradients
        grads = torch.autograd.grad(loss, adapted.parameters(), create_graph=not cfg.first_order)

        # Update parameters
        with torch.no_grad():
            for param, grad in zip(adapted.parameters(), grads, strict=False):
                param.sub_(cfg.inner_lr * grad)

    return adapted


class MAMLTrainer:
    """Trainer for MAML meta-learning."""

    def __init__(
        self,
        model: MAMLVAE,
        cfg: MAMLConfig,
        device: torch.device = None,
    ):
        self.model = model
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=cfg.outer_lr)

    def meta_train_step(
        self,
        tasks: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> dict[str, float]:
        """Perform one meta-training step.

        Args:
            tasks: List of (support_x, support_y, query_x, query_y) tuples

        Returns:
            Dict of metrics
        """
        self.model.train()
        self.meta_optimizer.zero_grad()

        meta_loss = 0.0
        query_losses = []

        for support_x, support_y, query_x, query_y in tasks:
            support_x = support_x.to(self.device)
            support_y = support_y.to(self.device)
            query_x = query_x.to(self.device)
            query_y = query_y.to(self.device)

            # Inner loop: adapt to support set
            adapted_model = adapt_model(self.model, self.cfg, support_x, support_y)

            # Outer loop: evaluate on query set
            query_loss = compute_task_loss(self.cfg, adapted_model, query_x, query_y)
            meta_loss += query_loss
            query_losses.append(query_loss.item())

        # Meta-update
        meta_loss = meta_loss / len(tasks)
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.meta_optimizer.step()

        return {
            "meta_loss": meta_loss.item(),
            "mean_query_loss": sum(query_losses) / len(query_losses),
        }

    def adapt_to_new_drug(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        n_steps: int | None = None,
    ) -> nn.Module:
        """Adapt model to a new drug with few examples.

        Args:
            support_x: Few-shot examples (e.g., 10 samples)
            support_y: Corresponding targets
            n_steps: Override number of adaptation steps

        Returns:
            Adapted model
        """
        self.model.eval()
        support_x = support_x.to(self.device)
        support_y = support_y.to(self.device)

        # Use more steps for final adaptation
        cfg = copy.copy(self.cfg)
        cfg.inner_steps = n_steps or self.cfg.inner_steps * 2

        return adapt_model(self.model, cfg, support_x, support_y)


class TaskSampler:
    """Sample tasks (drugs) for meta-learning."""

    def __init__(
        self,
        drug_data: dict[str, tuple[torch.Tensor, torch.Tensor]],
        n_support: int = 10,
        n_query: int = 20,
    ):
        """Initialize sampler.

        Args:
            drug_data: Dict mapping drug name to (X, y) tensors
            n_support: Number of support examples per task
            n_query: Number of query examples per task
        """
        self.drug_data = drug_data
        self.drug_names = list(drug_data.keys())
        self.n_support = n_support
        self.n_query = n_query

    def sample_task(self, drug: str | None = None) -> tuple[torch.Tensor, ...]:
        """Sample a single task.

        Returns:
            (support_x, support_y, query_x, query_y)
        """
        if drug is None:
            drug = self.drug_names[torch.randint(len(self.drug_names), (1,)).item()]

        X, y = self.drug_data[drug]
        n_total = X.size(0)

        # Random split
        indices = torch.randperm(n_total)
        support_idx = indices[: self.n_support]
        query_idx = indices[self.n_support : self.n_support + self.n_query]

        return (
            X[support_idx],
            y[support_idx],
            X[query_idx],
            y[query_idx],
        )

    def sample_batch(self, batch_size: int) -> list[tuple[torch.Tensor, ...]]:
        """Sample a batch of tasks."""
        return [self.sample_task() for _ in range(batch_size)]


class ProtoMAMLVAE(nn.Module):
    """Prototypical MAML: combines prototypes with meta-learning.

    Creates drug prototypes in latent space for faster adaptation.
    """

    def __init__(self, cfg: MAMLConfig, n_drugs: int):
        super().__init__()
        self.cfg = cfg
        self.n_drugs = n_drugs

        self.encoder = MAMLEncoder(cfg.input_dim, cfg.hidden_dims, cfg.latent_dim)
        self.decoder = MAMLDecoder(cfg.latent_dim, cfg.hidden_dims, cfg.input_dim)

        # Learnable drug prototypes
        self.drug_prototypes = nn.Parameter(torch.randn(n_drugs, cfg.latent_dim) * 0.1)

        # Prediction from prototype-conditioned latent
        self.predictor = nn.Sequential(
            nn.Linear(cfg.latent_dim * 2, cfg.latent_dim),
            nn.GELU(),
            nn.Linear(cfg.latent_dim, 1),
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        x: torch.Tensor,
        drug_idx: int,
    ) -> dict[str, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)

        # Condition on drug prototype
        prototype = self.drug_prototypes[drug_idx].unsqueeze(0).expand(x.size(0), -1)
        combined = torch.cat([z, prototype], dim=-1)
        prediction = self.predictor(combined).squeeze(-1)

        return {
            "x_recon": x_recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "prediction": prediction,
            "prototype": prototype,
        }

    def adapt_prototype(
        self,
        drug_idx: int,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        n_steps: int = 10,
        lr: float = 0.1,
    ):
        """Adapt drug prototype to new examples."""
        # Create learnable copy of prototype
        prototype = self.drug_prototypes[drug_idx].clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([prototype], lr=lr)

        for _ in range(n_steps):
            optimizer.zero_grad()

            mu, _ = self.encoder(support_x)
            combined = torch.cat([mu, prototype.unsqueeze(0).expand(mu.size(0), -1)], dim=-1)
            pred = self.predictor(combined).squeeze(-1)

            loss = F.mse_loss(pred, support_y)
            loss.backward()
            optimizer.step()

        # Update stored prototype
        with torch.no_grad():
            self.drug_prototypes[drug_idx] = prototype


def create_maml_vae(input_dim: int, variant: str = "standard", **kwargs) -> nn.Module:
    """Factory function for MAML VAE variants.

    Args:
        input_dim: Input dimension
        variant: 'standard' or 'proto'

    Returns:
        MAML VAE model
    """
    cfg = MAMLConfig(input_dim=input_dim, **kwargs)

    if variant == "standard":
        return MAMLVAE(cfg)
    elif variant == "proto":
        n_drugs = kwargs.get("n_drugs", 8)
        return ProtoMAMLVAE(cfg, n_drugs)
    else:
        raise ValueError(f"Unknown variant: {variant}")


if __name__ == "__main__":
    print("Testing MAML VAE")
    print("=" * 60)

    cfg = MAMLConfig(input_dim=99 * 22)
    model = MAMLVAE(cfg)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    x = torch.randn(8, 99 * 22)
    out = model(x)
    print(f"Prediction shape: {out['prediction'].shape}")

    # Test adaptation
    support_x = torch.randn(5, 99 * 22)
    support_y = torch.randn(5)

    adapted = adapt_model(model, cfg, support_x, support_y)
    out_adapted = adapted(x)
    print(f"Adapted prediction shape: {out_adapted['prediction'].shape}")

    # Test ProtoMAML
    proto_model = ProtoMAMLVAE(cfg, n_drugs=8)
    out_proto = proto_model(x, drug_idx=0)
    print(f"\nProtoMAML prediction shape: {out_proto['prediction'].shape}")
    print(f"Prototype shape: {out_proto['prototype'].shape}")

    print("\n" + "=" * 60)
    print("All MAML variants working!")
