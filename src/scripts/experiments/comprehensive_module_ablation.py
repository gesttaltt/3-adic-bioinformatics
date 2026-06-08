"""Comprehensive Module Ablation Study.

Tests all combinations of the 8 advanced modules to identify:
1. Individual module contributions
2. Synergistic combinations
3. Antagonistic combinations
4. Optimal configurations

Modules tested:
1. Hyperbolic geometry
2. Tropical geometry
3. P-adic triplet loss
4. P-adic ranking loss
5. Contrastive learning (simplified)
6. Different loss weight configurations

Output: Performance matrix and synergy analysis.
"""

from __future__ import annotations

import itertools
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class AblationConfig:
    """Configuration for ablation experiment."""

    input_dim: int = 9
    latent_dim: int = 16
    hidden_dims: list[int] = field(default_factory=lambda: [64, 32])
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Module flags
    use_hyperbolic: bool = False
    use_tropical: bool = False
    use_padic_triplet: bool = False
    use_padic_ranking: bool = False
    use_contrastive: bool = False

    # Loss weights
    padic_weight: float = 0.5
    ranking_weight: float = 0.3
    contrastive_weight: float = 0.1
    hyperbolic_curvature: float = 1.0
    tropical_temperature: float = 0.1

    def get_name(self) -> str:
        """Generate config name from flags."""
        parts = []
        if self.use_hyperbolic:
            parts.append("hyper")
        if self.use_tropical:
            parts.append("trop")
        if self.use_padic_triplet:
            parts.append("triplet")
        if self.use_padic_ranking:
            parts.append("rank")
        if self.use_contrastive:
            parts.append("contrast")
        return "_".join(parts) if parts else "baseline"


class AblationVAE(nn.Module):
    """VAE for ablation studies with modular components."""

    def __init__(self, config: AblationConfig):
        super().__init__()
        self.config = config

        # Build encoder
        encoder_layers = []
        in_dim = config.input_dim
        for h_dim in config.hidden_dims:
            encoder_layers.extend([nn.Linear(in_dim, h_dim), nn.ReLU(), nn.BatchNorm1d(h_dim)])
            in_dim = h_dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(in_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(in_dim, config.latent_dim)

        # Hyperbolic projection
        if config.use_hyperbolic:
            self.hyper_scale = nn.Parameter(torch.ones(1))

        # Tropical aggregation
        if config.use_tropical:
            self.tropical_heads = nn.Linear(config.latent_dim, 4)

        # Build decoder
        decoder_layers = []
        in_dim = config.latent_dim
        for h_dim in reversed(config.hidden_dims):
            decoder_layers.extend([nn.Linear(in_dim, h_dim), nn.ReLU(), nn.BatchNorm1d(h_dim)])
            in_dim = h_dim
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # Apply tropical smoothing
        if self.config.use_tropical:
            weights = F.softmax(self.tropical_heads(mu) / self.config.tropical_temperature, dim=-1)
            mu = mu * weights.mean(dim=-1, keepdim=True)

        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # Hyperbolic projection
        if self.config.use_hyperbolic:
            z_norm = torch.norm(z, dim=-1, keepdim=True)
            max_norm = (1.0 - 1e-5) / np.sqrt(self.config.hyperbolic_curvature)
            clamped = torch.clamp(z_norm, max=max_norm)
            z = z * (clamped / (z_norm + 1e-8)) * self.hyper_scale

        return z

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return {"x_recon": x_recon, "mu": mu, "logvar": logvar, "z": z}


class AblationLoss:
    """Loss computation for ablation study."""

    def __init__(self, config: AblationConfig):
        self.config = config

    def compute(
        self, output: dict[str, torch.Tensor], x: torch.Tensor, fitness: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        losses = {}

        # Reconstruction
        losses["recon"] = F.mse_loss(output["x_recon"], x)

        # KL divergence
        kl = -0.5 * torch.sum(1 + output["logvar"] - output["mu"].pow(2) - output["logvar"].exp())
        losses["kl"] = 0.001 * kl / x.size(0)

        z = output["z"]

        # P-adic triplet loss
        if self.config.use_padic_triplet:
            losses["triplet"] = self._padic_triplet(z, fitness) * self.config.padic_weight

        # P-adic ranking loss
        if self.config.use_padic_ranking:
            losses["ranking"] = self._padic_ranking(z, fitness) * self.config.ranking_weight

        # Contrastive loss
        if self.config.use_contrastive:
            losses["contrastive"] = self._contrastive(z) * self.config.contrastive_weight

        losses["total"] = sum(losses.values())
        return losses

    def _padic_triplet(self, z: torch.Tensor, fitness: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)
        if batch_size < 3:
            return torch.tensor(0.0, device=z.device)

        total_loss = 0.0
        n_triplets = 0

        for i in range(min(batch_size, 32)):  # Limit for speed
            fitness_diff = torch.abs(fitness - fitness[i])
            sorted_idx = torch.argsort(fitness_diff)

            if len(sorted_idx) >= 3:
                anchor = z[i]
                positive = z[sorted_idx[1]]
                negative = z[sorted_idx[-1]]

                d_pos = torch.norm(anchor - positive)
                d_neg = torch.norm(anchor - negative)

                triplet = F.relu(d_pos - d_neg + 1.0)
                total_loss += triplet
                n_triplets += 1

        return total_loss / max(n_triplets, 1)

    def _padic_ranking(self, z: torch.Tensor, fitness: torch.Tensor) -> torch.Tensor:
        z_proj = z[:, 0]
        z_centered = z_proj - z_proj.mean()
        f_centered = fitness - fitness.mean()

        z_std = torch.sqrt(torch.sum(z_centered**2) + 1e-8)
        f_std = torch.sqrt(torch.sum(f_centered**2) + 1e-8)

        corr = torch.sum(z_centered * f_centered) / (z_std * f_std)
        return -corr  # Maximize correlation

    def _contrastive(self, z: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        z_norm = F.normalize(z, dim=-1)
        sim = torch.mm(z_norm, z_norm.t()) / temperature
        labels = torch.arange(z.size(0), device=z.device)
        return F.cross_entropy(sim, labels)


def train_config(
    config: AblationConfig, train_x: torch.Tensor, train_labels: torch.Tensor, verbose: bool = False
) -> dict[str, float]:
    """Train with specific configuration and return metrics."""
    device = torch.device(config.device)

    model = AblationVAE(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = AblationLoss(config)

    dataset = TensorDataset(train_x, train_labels)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    best_corr = -1.0
    history_acc = []
    history_corr = []

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0

        for x, labels in dataloader:
            x = x.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(x)
            losses = loss_fn.compute(output, x, labels)
            losses["total"].backward()
            optimizer.step()
            epoch_loss += losses["total"].item()

        # Evaluate
        model.eval()
        with torch.no_grad():
            x_all = train_x.to(device)
            labels_all = train_labels.to(device)
            output = model(x_all)

            # Accuracy
            recon_error = F.mse_loss(output["x_recon"], x_all, reduction="none").mean(dim=-1)
            accuracy = (recon_error < 0.1).float().mean().item()

            # Correlation
            z_proj = output["z"][:, 0].cpu().numpy()
            labels_np = labels_all.cpu().numpy()

            if np.std(z_proj) > 1e-8 and np.std(labels_np) > 1e-8:
                correlation = np.corrcoef(z_proj, labels_np)[0, 1]
            else:
                correlation = 0.0

        history_acc.append(accuracy)
        history_corr.append(correlation)

        if correlation > best_corr:
            best_corr = correlation

        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1}: Acc={accuracy:.1%}, Corr={correlation:+.4f}")

    return {
        "best_correlation": best_corr,
        "final_correlation": history_corr[-1],
        "final_accuracy": history_acc[-1],
        "mean_correlation": np.mean(history_corr[-20:]),  # Last 20 epochs
    }


def generate_test_data(n_samples: int = 1000) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate test data with known structure."""
    np.random.seed(42)

    # Create structured data with hierarchical patterns
    sequences = np.zeros((n_samples, 9), dtype=np.float32)

    # Create clusters with fitness correlation
    n_clusters = 5
    cluster_centers = np.random.randn(n_clusters, 9).astype(np.float32)
    cluster_fitness = np.linspace(0, 1, n_clusters)

    labels = np.zeros(n_samples)
    for i in range(n_samples):
        cluster = np.random.randint(0, n_clusters)
        noise = np.random.randn(9).astype(np.float32) * 0.3
        sequences[i] = cluster_centers[cluster] + noise
        labels[i] = cluster_fitness[cluster] + np.random.randn() * 0.1

    # Normalize
    sequences = (sequences - sequences.min()) / (sequences.max() - sequences.min() + 1e-8)
    labels = (labels - labels.min()) / (labels.max() - labels.min() + 1e-8)

    return torch.tensor(sequences), torch.tensor(labels, dtype=torch.float32)


def run_ablation_study(epochs: int = 100, verbose: bool = True) -> pd.DataFrame:
    """Run comprehensive ablation study."""
    print("=" * 70)
    print("COMPREHENSIVE MODULE ABLATION STUDY")
    print("=" * 70)
    print()

    # Generate data
    print("Generating test data...")
    train_x, train_labels = generate_test_data(n_samples=1000)
    print(f"Data: {train_x.shape}, Labels: {train_labels.shape}")
    print()

    # Define all module combinations to test
    module_flags = ["use_hyperbolic", "use_tropical", "use_padic_triplet", "use_padic_ranking", "use_contrastive"]

    results = []

    # Test all 2^5 = 32 combinations
    total_experiments = 2 ** len(module_flags)
    print(f"Running {total_experiments} experiments...")
    print("-" * 70)

    for i, combo in enumerate(itertools.product([False, True], repeat=len(module_flags))):
        config = AblationConfig(epochs=epochs)

        # Set flags
        for flag, value in zip(module_flags, combo, strict=False):
            setattr(config, flag, value)

        name = config.get_name()

        if verbose:
            print(f"\n[{i + 1}/{total_experiments}] Testing: {name}")

        metrics = train_config(config, train_x, train_labels, verbose=False)

        result = {
            "name": name,
            "use_hyperbolic": config.use_hyperbolic,
            "use_tropical": config.use_tropical,
            "use_padic_triplet": config.use_padic_triplet,
            "use_padic_ranking": config.use_padic_ranking,
            "use_contrastive": config.use_contrastive,
            "n_modules": sum(combo),
            **metrics,
        }
        results.append(result)

        if verbose:
            print(f"  Result: Acc={metrics['final_accuracy']:.1%}, Corr={metrics['best_correlation']:+.4f}")

    df = pd.DataFrame(results)
    return df


def analyze_synergies(df: pd.DataFrame) -> dict[str, Any]:
    """Analyze module synergies from ablation results."""
    analysis = {}

    # Get baseline (no modules)
    baseline = df[df["n_modules"] == 0].iloc[0]
    analysis["baseline"] = {
        "accuracy": baseline["final_accuracy"],
        "correlation": baseline["best_correlation"],
    }

    # Individual module contributions
    individual = {}
    for col in ["use_hyperbolic", "use_tropical", "use_padic_triplet", "use_padic_ranking", "use_contrastive"]:
        # Get experiments with only this module
        single = df[(df[col]) & (df["n_modules"] == 1)]
        if len(single) > 0:
            module_name = col.replace("use_", "")
            individual[module_name] = {
                "accuracy_delta": single.iloc[0]["final_accuracy"] - baseline["final_accuracy"],
                "correlation_delta": single.iloc[0]["best_correlation"] - baseline["best_correlation"],
            }
    analysis["individual_contributions"] = individual

    # Find best combinations
    analysis["best_by_correlation"] = df.nlargest(5, "best_correlation")[
        ["name", "best_correlation", "final_accuracy", "n_modules"]
    ].to_dict("records")

    analysis["best_by_accuracy"] = df.nlargest(5, "final_accuracy")[
        ["name", "best_correlation", "final_accuracy", "n_modules"]
    ].to_dict("records")

    # Synergy analysis (pairs)
    synergies = []
    modules = ["hyperbolic", "tropical", "padic_triplet", "padic_ranking", "contrastive"]

    for m1, m2 in itertools.combinations(modules, 2):
        col1 = f"use_{m1}"
        col2 = f"use_{m2}"

        # Individual effects
        single1 = df[(df[col1]) & (df["n_modules"] == 1)]
        single2 = df[(df[col2]) & (df["n_modules"] == 1)]

        # Combined effect
        combo = df[(df[col1]) & (df[col2]) & (df["n_modules"] == 2)]

        if len(single1) > 0 and len(single2) > 0 and len(combo) > 0:
            expected = (
                single1.iloc[0]["best_correlation"] + single2.iloc[0]["best_correlation"] - baseline["best_correlation"]
            )
            actual = combo.iloc[0]["best_correlation"]
            synergy = actual - expected

            synergies.append(
                {
                    "modules": f"{m1} + {m2}",
                    "expected": expected,
                    "actual": actual,
                    "synergy": synergy,
                    "synergy_type": "positive" if synergy > 0.01 else ("negative" if synergy < -0.01 else "neutral"),
                }
            )

    analysis["synergies"] = sorted(synergies, key=lambda x: x["synergy"], reverse=True)

    return analysis


def print_analysis(df: pd.DataFrame, analysis: dict[str, Any]):
    """Print formatted analysis results."""
    print()
    print("=" * 70)
    print("ABLATION STUDY RESULTS")
    print("=" * 70)

    print("\n1. BASELINE PERFORMANCE (no modules)")
    print(f"   Accuracy: {analysis['baseline']['accuracy']:.1%}")
    print(f"   Correlation: {analysis['baseline']['correlation']:+.4f}")

    print("\n2. INDIVIDUAL MODULE CONTRIBUTIONS")
    print("-" * 70)
    print(f"{'Module':<20} {'Accuracy Delta':<18} {'Correlation Delta':<18}")
    print("-" * 70)
    for module, contrib in analysis["individual_contributions"].items():
        print(f"{module:<20} {contrib['accuracy_delta']:+.1%}             {contrib['correlation_delta']:+.4f}")

    print("\n3. BEST CONFIGURATIONS BY CORRELATION")
    print("-" * 70)
    print(f"{'Rank':<6} {'Configuration':<30} {'Correlation':<15} {'Accuracy':<15}")
    print("-" * 70)
    for i, config in enumerate(analysis["best_by_correlation"], 1):
        print(f"{i:<6} {config['name']:<30} {config['best_correlation']:+.4f}         {config['final_accuracy']:.1%}")

    print("\n4. BEST CONFIGURATIONS BY ACCURACY")
    print("-" * 70)
    print(f"{'Rank':<6} {'Configuration':<30} {'Accuracy':<15} {'Correlation':<15}")
    print("-" * 70)
    for i, config in enumerate(analysis["best_by_accuracy"], 1):
        print(f"{i:<6} {config['name']:<30} {config['final_accuracy']:.1%}           {config['best_correlation']:+.4f}")

    print("\n5. MODULE SYNERGIES (pairs)")
    print("-" * 70)
    print(f"{'Combination':<30} {'Expected':<12} {'Actual':<12} {'Synergy':<12} {'Type':<10}")
    print("-" * 70)
    for syn in analysis["synergies"]:
        print(
            f"{syn['modules']:<30} {syn['expected']:+.4f}      {syn['actual']:+.4f}      "
            f"{syn['synergy']:+.4f}      {syn['synergy_type']:<10}"
        )

    # Summary insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)

    # Best individual
    best_individual = max(analysis["individual_contributions"].items(), key=lambda x: x[1]["correlation_delta"])
    print(
        f"• Best individual module for correlation: {best_individual[0]} ({best_individual[1]['correlation_delta']:+.4f})"
    )

    # Best synergy
    if analysis["synergies"]:
        best_synergy = analysis["synergies"][0]
        if best_synergy["synergy"] > 0.01:
            print(f"• Strongest positive synergy: {best_synergy['modules']} (+{best_synergy['synergy']:.4f})")

    # Best overall
    best_overall = analysis["best_by_correlation"][0]
    print(f"• Best overall configuration: {best_overall['name']} (corr={best_overall['best_correlation']:+.4f})")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive Module Ablation Study")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output", type=str, default="ablation_results.json")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # Run ablation study
    df = run_ablation_study(epochs=args.epochs, verbose=not args.quiet)

    # Analyze synergies
    analysis = analyze_synergies(df)

    # Print results
    print_analysis(df, analysis)

    # Save results
    output_path = Path(project_root) / "results" / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_data = {
        "timestamp": datetime.now().isoformat(),
        "epochs": args.epochs,
        "experiments": df.to_dict("records"),
        "analysis": analysis,
    }

    with open(output_path, "w") as f:
        json.dump(results_data, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
