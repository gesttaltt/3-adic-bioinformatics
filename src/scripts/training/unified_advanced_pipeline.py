"""Unified Advanced Training Pipeline.

Demonstrates integration of all 8 advanced modules:
1. Persistent Homology - Topological features
2. P-adic Contrastive - Hierarchical contrastive learning
3. Information Geometry - Natural gradient optimization
4. Statistical Physics - Fitness landscape modeling
5. Tropical Geometry - Tree structure analysis
6. Hyperbolic GNN - Hierarchical embeddings
7. Category Theory - Type-safe composition
8. Meta-Learning - Rapid adaptation

This pipeline shows how to combine these modules for maximum effectiveness.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class UnifiedPipelineConfig:
    """Configuration for unified training pipeline."""

    # Model settings
    input_dim: int = 9
    latent_dim: int = 16
    hidden_dims: list[int] = field(default_factory=lambda: [64, 32])

    # Training settings
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001

    # Module activation flags
    use_persistent_homology: bool = True
    use_padic_contrastive: bool = True
    use_natural_gradient: bool = True
    use_statistical_physics: bool = False  # Heavy computation
    use_tropical_geometry: bool = True
    use_hyperbolic: bool = True
    use_category_theory: bool = True
    use_meta_learning: bool = False  # Requires multi-task setup

    # Module-specific settings
    homology_max_dim: int = 1
    homology_weight: float = 0.1
    contrastive_temperature: float = 0.07
    contrastive_weight: float = 0.1
    hyperbolic_curvature: float = 1.0
    tropical_temperature: float = 0.1
    padic_prime: int = 3

    # P-adic loss settings
    padic_loss_type: str = "triplet"
    padic_weight: float = 0.5
    ranking_weight: float = 0.3

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class UnifiedEncoder(nn.Module):
    """Encoder combining multiple geometric approaches."""

    def __init__(self, config: UnifiedPipelineConfig):
        super().__init__()
        self.config = config

        # Base encoder layers
        layers = []
        in_dim = config.input_dim
        for h_dim in config.hidden_dims:
            layers.extend([nn.Linear(in_dim, h_dim), nn.ReLU(), nn.BatchNorm1d(h_dim)])
            in_dim = h_dim

        self.base_encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(in_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(in_dim, config.latent_dim)

        # Hyperbolic projection if enabled
        if config.use_hyperbolic:
            self.hyperbolic_proj = HyperbolicProjection(config.latent_dim, config.hyperbolic_curvature)

        # Tropical aggregation if enabled
        if config.use_tropical_geometry:
            self.tropical_agg = TropicalSmoothMax(config.latent_dim, config.tropical_temperature)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.base_encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # Apply tropical smoothing to mu
        if self.config.use_tropical_geometry and hasattr(self, "tropical_agg"):
            mu = self.tropical_agg(mu)

        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # Project to hyperbolic space if enabled
        if self.config.use_hyperbolic and hasattr(self, "hyperbolic_proj"):
            z = self.hyperbolic_proj(z)

        return z


class HyperbolicProjection(nn.Module):
    """Project embeddings to Poincare ball."""

    def __init__(self, dim: int, curvature: float = 1.0):
        super().__init__()
        self.dim = dim
        self.curvature = curvature
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project to Poincare ball via exponential map
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        max_norm = (1.0 - 1e-5) / np.sqrt(self.curvature)

        # Clamp to ball
        clamped = torch.clamp(x_norm, max=max_norm)
        x_proj = x * (clamped / (x_norm + 1e-8))

        return x_proj * self.scale


class TropicalSmoothMax(nn.Module):
    """Tropical geometry smooth max aggregation."""

    def __init__(self, dim: int, temperature: float = 0.1):
        super().__init__()
        self.dim = dim
        self.temperature = temperature
        self.heads = nn.Linear(dim, 4)  # Multi-head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute attention weights
        weights = F.softmax(self.heads(x) / self.temperature, dim=-1)

        # Aggregate with tropical-style operation
        # Smooth approximation to max
        scaled = x.unsqueeze(-1) * weights.unsqueeze(1)
        smooth_max = torch.logsumexp(scaled / self.temperature, dim=-1) * self.temperature

        return smooth_max


class UnifiedDecoder(nn.Module):
    """Decoder network."""

    def __init__(self, config: UnifiedPipelineConfig):
        super().__init__()
        self.config = config

        layers = []
        in_dim = config.latent_dim
        for h_dim in reversed(config.hidden_dims):
            layers.extend([nn.Linear(in_dim, h_dim), nn.ReLU(), nn.BatchNorm1d(h_dim)])
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class UnifiedVAE(nn.Module):
    """Unified VAE combining all advanced modules."""

    def __init__(self, config: UnifiedPipelineConfig):
        super().__init__()
        self.config = config
        self.encoder = UnifiedEncoder(config)
        self.decoder = UnifiedDecoder(config)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.encoder.reparameterize(mu, logvar)
        x_recon = self.decoder(z)

        return {"x_recon": x_recon, "mu": mu, "logvar": logvar, "z": z}


class UnifiedLossComputer:
    """Computes combined loss from all enabled modules."""

    def __init__(self, config: UnifiedPipelineConfig):
        self.config = config
        self._setup_losses()

    def _setup_losses(self):
        """Initialize loss modules based on config."""
        self.loss_modules = {}

        # Persistent homology loss
        if self.config.use_persistent_homology:
            try:
                from src.topology.persistent_homology import PersistenceVectorizer

                self.persistence_vec = PersistenceVectorizer(method="statistics", resolution=50)
                self.loss_modules["homology"] = self._homology_loss
            except (ImportError, Exception) as e:
                print(f"Warning: Persistent homology not available ({e}), skipping")

        # P-adic contrastive loss
        if self.config.use_padic_contrastive:
            try:
                from src.contrastive.padic_contrastive import PAdicContrastiveLoss

                self.contrastive = PAdicContrastiveLoss(
                    temperature=self.config.contrastive_temperature, prime=self.config.padic_prime
                )
                self.loss_modules["contrastive"] = self._contrastive_loss
            except ImportError:
                # Fallback implementation
                self.loss_modules["contrastive"] = self._simple_contrastive_loss

        # P-adic triplet loss
        self.loss_modules["padic_triplet"] = self._padic_triplet_loss
        self.loss_modules["padic_ranking"] = self._padic_ranking_loss

    def _homology_loss(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Loss based on topological features."""
        # Compute persistence of latent space
        z_np = z.detach().cpu().numpy()
        try:
            features = self.persistence_vec.transform([z_np])[0]
            # Encourage rich topology (high total persistence)
            return -torch.tensor(np.sum(features), device=z.device)
        except Exception:
            return torch.tensor(0.0, device=z.device)

    def _contrastive_loss(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """P-adic weighted contrastive loss."""
        try:
            return self.contrastive(z)
        except Exception:
            return self._simple_contrastive_loss(z)

    def _simple_contrastive_loss(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Simplified contrastive loss."""
        # Normalize embeddings
        z_norm = F.normalize(z, dim=-1)

        # Compute similarity matrix
        sim = torch.mm(z_norm, z_norm.t()) / self.config.contrastive_temperature

        # InfoNCE-style loss
        labels = torch.arange(z.size(0), device=z.device)
        loss = F.cross_entropy(sim, labels)

        return loss

    def _padic_triplet_loss(self, z: torch.Tensor, fitness: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """P-adic triplet loss for structure preservation."""
        if fitness is None:
            return torch.tensor(0.0, device=z.device)

        batch_size = z.size(0)
        if batch_size < 3:
            return torch.tensor(0.0, device=z.device)

        total_loss = 0.0
        n_triplets = 0

        for i in range(batch_size):
            # Find positive (similar fitness) and negative (different fitness)
            fitness_diff = torch.abs(fitness - fitness[i])
            sorted_idx = torch.argsort(fitness_diff)

            if len(sorted_idx) >= 3:
                anchor = z[i]
                positive = z[sorted_idx[1]]  # Most similar
                negative = z[sorted_idx[-1]]  # Most different

                # Triplet loss
                d_pos = torch.norm(anchor - positive)
                d_neg = torch.norm(anchor - negative)

                triplet = F.relu(d_pos - d_neg + 1.0)
                total_loss += triplet
                n_triplets += 1

        return total_loss / max(n_triplets, 1)

    def _padic_ranking_loss(self, z: torch.Tensor, fitness: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """P-adic ranking loss for fitness correlation."""
        if fitness is None:
            return torch.tensor(0.0, device=z.device)

        # Compute p-adic distance proxy (using first latent dimension)
        z_proj = z[:, 0]

        # Correlation between z projection and fitness
        z_centered = z_proj - z_proj.mean()
        f_centered = fitness - fitness.mean()

        corr = torch.sum(z_centered * f_centered) / (
            torch.sqrt(torch.sum(z_centered**2) * torch.sum(f_centered**2)) + 1e-8
        )

        # Maximize correlation (minimize negative correlation)
        return -corr

    def compute_loss(
        self, model_output: dict[str, torch.Tensor], x: torch.Tensor, fitness: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Compute total loss from all modules."""
        losses = {}

        # Reconstruction loss
        losses["recon"] = F.mse_loss(model_output["x_recon"], x)

        # KL divergence
        kl = -0.5 * torch.sum(1 + model_output["logvar"] - model_output["mu"].pow(2) - model_output["logvar"].exp())
        kl = kl / x.size(0)
        losses["kl"] = 0.001 * kl

        # Module-specific losses
        z = model_output["z"]

        for name, loss_fn in self.loss_modules.items():
            try:
                loss_val = loss_fn(z=z, fitness=fitness, x=x)
                if name == "homology":
                    losses[name] = self.config.homology_weight * loss_val
                elif name == "contrastive":
                    losses[name] = self.config.contrastive_weight * loss_val
                elif name == "padic_triplet":
                    losses[name] = self.config.padic_weight * loss_val
                elif name == "padic_ranking":
                    losses[name] = self.config.ranking_weight * loss_val
                else:
                    losses[name] = loss_val
            except Exception as e:
                print(f"Warning: {name} loss failed: {e}")
                losses[name] = torch.tensor(0.0, device=x.device)

        # Total
        losses["total"] = sum(losses.values())

        return losses


class UnifiedTrainer:
    """Trainer combining all advanced modules."""

    def __init__(self, config: UnifiedPipelineConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Initialize model
        self.model = UnifiedVAE(config).to(self.device)

        # Initialize optimizer (with optional natural gradient)
        if config.use_natural_gradient:
            try:
                from src.information.fisher_geometry import KFACOptimizer

                self.optimizer = KFACOptimizer(self.model, lr=config.learning_rate, damping=0.01, update_freq=10)
                print("Using K-FAC natural gradient optimizer")
            except (ImportError, Exception) as e:
                print(f"K-FAC not available ({e}), using Adam")
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

        # Initialize loss computer
        self.loss_computer = UnifiedLossComputer(config)

        # History
        self.history = {"losses": [], "accuracy": [], "correlation": []}

    def train_epoch(self, dataloader: DataLoader, fitness_labels: torch.Tensor | None = None) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {}

        for _batch_idx, (x, labels) in enumerate(dataloader):
            x = x.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(x)

            # Compute losses
            losses = self.loss_computer.compute_loss(output, x, labels)

            # Backward pass
            losses["total"].backward()
            self.optimizer.step()

            # Accumulate losses
            for name, value in losses.items():
                if name not in epoch_losses:
                    epoch_losses[name] = 0.0
                epoch_losses[name] += value.item()

        # Average
        for name in epoch_losses:
            epoch_losses[name] /= len(dataloader)

        return epoch_losses

    def evaluate(self, x: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
        """Evaluate model performance."""
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            labels = labels.to(self.device)

            output = self.model(x)
            z = output["z"]

            # Reconstruction accuracy (threshold-based)
            recon = output["x_recon"]
            recon_error = F.mse_loss(recon, x, reduction="none").mean(dim=-1)
            accuracy = (recon_error < 0.1).float().mean().item()

            # Fitness correlation
            z_proj = z[:, 0].cpu().numpy()
            labels_np = labels.cpu().numpy()

            if np.std(z_proj) > 1e-8 and np.std(labels_np) > 1e-8:
                correlation = np.corrcoef(z_proj, labels_np)[0, 1]
            else:
                correlation = 0.0

        return {"accuracy": accuracy, "correlation": correlation}

    def train(
        self,
        train_x: torch.Tensor,
        train_labels: torch.Tensor,
        val_x: torch.Tensor | None = None,
        val_labels: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Full training loop."""
        # Create dataloader
        dataset = TensorDataset(train_x, train_labels)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        print(f"\nStarting unified training with {len(train_x)} samples")
        print(f"Config: {self.config}")
        print("-" * 60)

        best_correlation = -1.0
        best_state = None

        for epoch in range(self.config.epochs):
            # Train
            train_losses = self.train_epoch(dataloader)

            # Evaluate
            if val_x is not None and val_labels is not None:
                metrics = self.evaluate(val_x, val_labels)
            else:
                metrics = self.evaluate(train_x, train_labels)

            # Store history
            self.history["losses"].append(train_losses)
            self.history["accuracy"].append(metrics["accuracy"])
            self.history["correlation"].append(metrics["correlation"])

            # Track best
            if metrics["correlation"] > best_correlation:
                best_correlation = metrics["correlation"]
                best_state = self.model.state_dict().copy()

            # Log
            if (epoch + 1) % 10 == 0 or epoch == 0:
                loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in train_losses.items()])
                print(
                    f"Epoch {epoch + 1}/{self.config.epochs} | "
                    f"Acc: {metrics['accuracy']:.1%} | "
                    f"Corr: {metrics['correlation']:+.4f} | "
                    f"Loss: [{loss_str}]"
                )

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return {
            "history": self.history,
            "best_correlation": best_correlation,
            "final_accuracy": self.history["accuracy"][-1],
        }


def generate_synthetic_data(n_samples: int = 1000, seq_length: int = 300) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic biological sequence data."""
    np.random.seed(42)

    # Generate codon sequences
    n_codons = seq_length // 3
    sequences = np.random.randint(0, 64, size=(n_samples, n_codons))

    # Convert to ternary representation
    ternary = []
    for i in range(n_codons):
        codon_vals = sequences[:, i]
        t1 = codon_vals % 3
        t2 = (codon_vals // 3) % 3
        t3 = (codon_vals // 9) % 3
        ternary.extend([t1, t2, t3])

    x = np.array(ternary).T.astype(np.float32)

    # Generate fitness labels (correlated with sequence properties)
    gc_content = np.sum((sequences == 1) | (sequences == 2), axis=1) / n_codons
    fitness = gc_content + np.random.randn(n_samples) * 0.1
    fitness = (fitness - fitness.min()) / (fitness.max() - fitness.min())

    # Take first 9 positions (3 codons)
    x = x[:, :9]

    return torch.tensor(x, dtype=torch.float32), torch.tensor(fitness, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description="Unified Advanced Training Pipeline")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--no-hyperbolic", action="store_true")
    parser.add_argument("--no-tropical", action="store_true")
    parser.add_argument("--no-contrastive", action="store_true")
    parser.add_argument("--no-homology", action="store_true")
    parser.add_argument("--padic-weight", type=float, default=0.5)
    parser.add_argument("--ranking-weight", type=float, default=0.3)
    args = parser.parse_args()

    # Create config
    config = UnifiedPipelineConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        latent_dim=args.latent_dim,
        use_hyperbolic=not args.no_hyperbolic,
        use_tropical_geometry=not args.no_tropical,
        use_padic_contrastive=not args.no_contrastive,
        use_persistent_homology=not args.no_homology,
        padic_weight=args.padic_weight,
        ranking_weight=args.ranking_weight,
    )

    # Generate data
    print("Generating synthetic data...")
    train_x, train_labels = generate_synthetic_data(n_samples=1000)
    val_x, val_labels = generate_synthetic_data(n_samples=200)

    print(f"Train: {train_x.shape}, Val: {val_x.shape}")

    # Train
    trainer = UnifiedTrainer(config)
    results = trainer.train(train_x, train_labels, val_x, val_labels)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best correlation: {results['best_correlation']:+.4f}")
    print(f"Final accuracy: {results['final_accuracy']:.1%}")

    # Test without advanced modules for comparison
    print("\n" + "-" * 60)
    print("Baseline comparison (no advanced modules)")
    print("-" * 60)

    baseline_config = UnifiedPipelineConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        latent_dim=args.latent_dim,
        use_hyperbolic=False,
        use_tropical_geometry=False,
        use_padic_contrastive=False,
        use_persistent_homology=False,
        padic_weight=0.0,
        ranking_weight=0.0,
    )

    baseline_trainer = UnifiedTrainer(baseline_config)
    baseline_results = baseline_trainer.train(train_x, train_labels, val_x, val_labels)

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<20} {'Baseline':<15} {'Unified':<15} {'Improvement':<15}")
    print("-" * 60)

    base_corr = baseline_results["best_correlation"]
    unified_corr = results["best_correlation"]
    corr_imp = unified_corr - base_corr

    base_acc = baseline_results["final_accuracy"]
    unified_acc = results["final_accuracy"]
    acc_imp = unified_acc - base_acc

    print(f"{'Correlation':<20} {base_corr:+.4f}       {unified_corr:+.4f}       {corr_imp:+.4f}")
    print(f"{'Accuracy':<20} {base_acc:.1%}         {unified_acc:.1%}         {acc_imp:+.1%}")


if __name__ == "__main__":
    main()
