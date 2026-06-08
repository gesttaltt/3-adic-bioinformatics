#!/usr/bin/env python
"""Test Alternative P-Adic Loss Implementations.

This script tests different approaches to p-adic structure preservation:
1. Current triplet ranking (baseline - known to fail)
2. Soft ranking with KL divergence
3. Contrastive learning with p-adic similarity
4. Multi-scale per-valuation loss
5. Dual-head architecture

Each approach is evaluated on:
- Spearman correlation with p-adic distances
- Reconstruction accuracy
- Training stability

Run with: python src/scripts/experiments/test_alternative_padic.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def compute_padic_distance(i: int, j: int, p: int = 3) -> float:
    """Compute p-adic distance."""
    if i == j:
        return 0.0
    diff = abs(i - j)
    k = 0
    while diff % p == 0:
        diff //= p
        k += 1
    return float(p ** (-k))


def compute_padic_matrix(n: int) -> torch.Tensor:
    """Precompute p-adic distance matrix."""
    matrix = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            matrix[i, j] = compute_padic_distance(i, j)
    return matrix


def evaluate_structure(z: torch.Tensor, indices: torch.Tensor) -> dict[str, float]:
    """Evaluate latent space structure."""
    n = len(z)
    n_pairs = min(3000, n * (n - 1) // 2)

    np.random.seed(42)
    i_idx = np.random.randint(0, n, n_pairs)
    j_idx = np.random.randint(0, n, n_pairs)
    mask = i_idx != j_idx
    i_idx, j_idx = i_idx[mask], j_idx[mask]

    padic_dists = [
        compute_padic_distance(indices[i].item(), indices[j].item()) for i, j in zip(i_idx, j_idx, strict=False)
    ]
    latent_dists = torch.norm(z[i_idx] - z[j_idx], dim=1).detach().numpy()

    corr, pval = spearmanr(padic_dists, latent_dists)

    return {
        "spearman": corr if not np.isnan(corr) else 0.0,
        "pvalue": pval if not np.isnan(pval) else 1.0,
    }


# =============================================================================
# ALTERNATIVE P-ADIC LOSS IMPLEMENTATIONS
# =============================================================================


class SoftPadicRankingLoss(nn.Module):
    """Soft p-adic ranking using KL divergence between soft rankings.

    Instead of hard triplet constraints, this uses softmax to create
    probability distributions over "which points are close" and matches
    them via KL divergence.

    Advantages:
    - Smooth gradients (no margin discontinuity)
    - Considers all pairs, not just sampled triplets
    - More stable optimization
    """

    def __init__(self, temperature: float = 0.5, n_samples: int = 200):
        super().__init__()
        self.temperature = temperature
        self.n_samples = n_samples

    def forward(self, z: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        n = z.size(0)
        if n < 3:
            return torch.tensor(0.0, device=z.device)

        # Subsample for efficiency
        if n > self.n_samples:
            idx = torch.randperm(n)[: self.n_samples]
            z = z[idx]
            indices = indices[idx]
            n = self.n_samples

        # Compute latent distance matrix
        latent_dist = torch.cdist(z, z)  # (n, n)

        # Compute p-adic distance matrix
        padic_dist = torch.zeros(n, n, device=z.device)
        for i in range(n):
            for j in range(n):
                padic_dist[i, j] = compute_padic_distance(indices[i].item(), indices[j].item())

        # Convert to soft rankings (probability that j is "near" i)
        # Lower distance = higher probability
        latent_ranks = F.softmax(-latent_dist / self.temperature, dim=1)
        padic_ranks = F.softmax(-padic_dist / self.temperature, dim=1)

        # KL divergence: how different are the rankings?
        # We want latent rankings to match p-adic rankings
        loss = F.kl_div(latent_ranks.log(), padic_ranks, reduction="batchmean")

        return loss


class ContrastivePadicLoss(nn.Module):
    """InfoNCE-style contrastive loss with p-adic similarity targets.

    Uses normalized embeddings and cross-entropy to match similarity
    distributions to p-adic-based targets.

    Advantages:
    - Scale-invariant (normalized embeddings)
    - Proven effective in contrastive learning literature
    - Smooth optimization landscape
    """

    def __init__(self, temperature: float = 0.1, n_samples: int = 200):
        super().__init__()
        self.temperature = temperature
        self.n_samples = n_samples

    def forward(self, z: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        n = z.size(0)
        if n < 3:
            return torch.tensor(0.0, device=z.device)

        # Subsample for efficiency
        if n > self.n_samples:
            idx = torch.randperm(n)[: self.n_samples]
            z = z[idx]
            indices = indices[idx]
            n = self.n_samples

        # Normalize embeddings
        z_norm = F.normalize(z, dim=1)

        # Similarity matrix (cosine similarity)
        sim = torch.mm(z_norm, z_norm.t()) / self.temperature

        # Target similarity from p-adic distance
        # Close in p-adic = high similarity
        target_sim = torch.zeros(n, n, device=z.device)
        for i in range(n):
            for j in range(n):
                d = compute_padic_distance(indices[i].item(), indices[j].item())
                # Convert distance to similarity (inverse relationship)
                # d=0 -> sim=1, d=1 -> sim=0.5, d=1/3 -> sim=0.75
                target_sim[i, j] = 1.0 / (1.0 + d)

        # Normalize targets to probabilities
        target_probs = F.softmax(target_sim / 0.1, dim=1)

        # Cross-entropy loss
        log_probs = F.log_softmax(sim, dim=1)
        loss = -(target_probs * log_probs).sum(dim=1).mean()

        return loss


class MultiscalePadicLoss(nn.Module):
    """Multi-scale p-adic loss targeting different valuation levels.

    Instead of a single loss, this creates separate constraints for each
    valuation level (0, 1, 2, ...), with distance targets that decrease
    exponentially.

    Advantages:
    - Explicit scale for each hierarchy level
    - Matches the exponential structure of p-adic distances
    - More interpretable
    """

    def __init__(self, max_valuation: int = 5, n_samples: int = 200):
        super().__init__()
        self.max_valuation = max_valuation
        self.n_samples = n_samples

    def forward(self, z: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        n = z.size(0)
        if n < 3:
            return torch.tensor(0.0, device=z.device)

        # Subsample
        if n > self.n_samples:
            idx = torch.randperm(n)[: self.n_samples]
            z = z[idx]
            indices = indices[idx]
            n = self.n_samples

        total_loss = torch.tensor(0.0, device=z.device)
        level_count = 0

        # Compute all pairwise distances once
        latent_dist = torch.cdist(z, z)

        # For each valuation level
        for v in range(self.max_valuation + 1):
            # Target distance for this level
            target = 3.0 ** (-v)

            # Find pairs with this valuation
            pairs_i = []
            pairs_j = []
            for i in range(n):
                for j in range(i + 1, n):
                    diff = abs(indices[i].item() - indices[j].item())
                    if diff > 0:
                        k = 0
                        temp = diff
                        while temp % 3 == 0:
                            temp //= 3
                            k += 1
                        if k == v:
                            pairs_i.append(i)
                            pairs_j.append(j)

            if len(pairs_i) < 2:
                continue

            # Get actual distances for these pairs
            pairs_i = torch.tensor(pairs_i, device=z.device)
            pairs_j = torch.tensor(pairs_j, device=z.device)
            actual_dists = latent_dist[pairs_i, pairs_j]

            # MSE to target (scaled by level importance)
            weight = 1.0 / (v + 1)  # Higher valuation = rarer, more important
            level_loss = weight * F.mse_loss(actual_dists, torch.full_like(actual_dists, target))
            total_loss = total_loss + level_loss
            level_count += 1

        return total_loss / max(level_count, 1)


class GradientAlignedPadicLoss(nn.Module):
    """P-adic loss that aligns with reconstruction gradients.

    Instead of fighting reconstruction, this loss only activates when
    the reconstruction gradient and p-adic gradient are aligned.

    Uses gradient projection to ensure no conflict.
    """

    def __init__(self, base_loss: nn.Module = None, alignment_threshold: float = 0.0):
        super().__init__()
        self.base_loss = base_loss or SoftPadicRankingLoss()
        self.threshold = alignment_threshold

    def forward(self, z: torch.Tensor, indices: torch.Tensor, recon_grad: torch.Tensor = None) -> torch.Tensor:
        """Compute aligned p-adic loss.

        Args:
            z: Embeddings
            indices: Operation indices
            recon_grad: Gradient from reconstruction loss (if available)
        """
        # Compute base p-adic loss
        padic_loss = self.base_loss(z, indices)

        if recon_grad is None:
            return padic_loss

        # Compute p-adic gradient
        padic_grad = torch.autograd.grad(padic_loss, z, retain_graph=True)[0]

        # Project p-adic gradient onto reconstruction gradient
        # Only keep component that doesn't fight reconstruction
        cos_sim = F.cosine_similarity(padic_grad.flatten().unsqueeze(0), recon_grad.flatten().unsqueeze(0))

        if cos_sim < self.threshold:
            # Gradients conflict - reduce p-adic contribution
            scale = max(0, (cos_sim + 1) / 2)  # Maps [-1, 1] to [0, 1]
            return scale * padic_loss

        return padic_loss


# =============================================================================
# DUAL-HEAD VAE
# =============================================================================


class DualHeadVAE(nn.Module):
    """VAE with separate head for p-adic structure.

    The key idea: use a separate projection head for p-adic structure,
    so reconstruction and p-adic objectives don't compete.

    Architecture:
        x -> Encoder -> z (shared) -> Decoder -> recon
                           |
                           -> P-adic Head -> z_padic
    """

    def __init__(self, input_dim: int = 9, latent_dim: int = 16, hidden_dims: list = None, padic_dim: int = 16):
        super().__init__()
        hidden_dims = hidden_dims or [64, 32]

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.ReLU(),
                ]
            )
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Latent projections
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.ReLU(),
                ]
            )
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim * 3))
        self.decoder = nn.Sequential(*decoder_layers)

        # P-adic head (separate from decoder!)
        self.padic_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, padic_dim),
        )

        self.input_dim = input_dim
        self.latent_dim = latent_dim

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> dict:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)

        # Reconstruction path
        logits = self.decoder(z).view(-1, self.input_dim, 3)

        # P-adic path (separate!)
        z_padic = self.padic_head(z)

        return {
            "logits": logits,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "z_padic": z_padic,  # Use this for p-adic loss
        }


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================


@dataclass
class ExperimentResult:
    name: str
    spearman: float
    accuracy: float
    training_time: float
    final_loss: float
    details: dict = None


def run_experiment(
    name: str,
    model_class,
    padic_loss_class,
    use_padic_head: bool = False,
    padic_weight: float = 0.3,
    epochs: int = 50,
) -> ExperimentResult:
    """Run a single experiment configuration."""
    from src.data.generation import generate_all_ternary_operations
    from src.losses.dual_vae_loss import KLDivergenceLoss, ReconstructionLoss

    print(f"\n  Running: {name}...")
    start_time = time.time()

    # Load data
    ops = torch.tensor(generate_all_ternary_operations()[:500], dtype=torch.float32)
    indices = torch.arange(len(ops))

    # Create model
    model = model_class()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Loss functions
    recon_fn = ReconstructionLoss()
    kl_fn = KLDivergenceLoss()
    padic_fn = padic_loss_class() if padic_loss_class else None

    # Training
    for _epoch in range(epochs):
        model.train()
        outputs = model(ops)

        # Reconstruction loss
        recon = recon_fn(outputs["logits"], ops)
        kl = kl_fn(outputs["mu"], outputs["logvar"])
        loss = recon + 0.01 * kl

        # P-adic loss
        if padic_fn:
            z_for_padic = outputs["z_padic"] if use_padic_head else outputs["z"]
            padic = padic_fn(z_for_padic, indices)
            loss = loss + padic_weight * padic

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    training_time = time.time() - start_time

    # Evaluate
    model.eval()
    with torch.no_grad():
        outputs = model(ops)
        z = outputs["z"]

        # Accuracy
        pred = outputs["logits"].argmax(dim=-1)
        target = (ops + 1).long()
        accuracy = (pred == target).float().mean().item()

        # Structure
        metrics = evaluate_structure(z, indices)

    return ExperimentResult(
        name=name,
        spearman=metrics["spearman"],
        accuracy=accuracy,
        training_time=training_time,
        final_loss=loss.item(),
    )


def main():
    from src.models.simple_vae import SimpleVAE, SimpleVAEWithHyperbolic

    print("\n" + "=" * 70)
    print("ALTERNATIVE P-ADIC IMPLEMENTATION TESTS")
    print("=" * 70)

    results = []

    # Baseline experiments
    print("\n--- Baselines ---")

    results.append(
        run_experiment(
            name="baseline_no_padic",
            model_class=SimpleVAE,
            padic_loss_class=None,
        )
    )

    results.append(
        run_experiment(
            name="hyperbolic_no_padic",
            model_class=SimpleVAEWithHyperbolic,
            padic_loss_class=None,
        )
    )

    # Alternative p-adic losses
    print("\n--- Alternative P-adic Losses ---")

    results.append(
        run_experiment(
            name="soft_ranking",
            model_class=SimpleVAEWithHyperbolic,
            padic_loss_class=SoftPadicRankingLoss,
        )
    )

    results.append(
        run_experiment(
            name="contrastive",
            model_class=SimpleVAEWithHyperbolic,
            padic_loss_class=ContrastivePadicLoss,
        )
    )

    results.append(
        run_experiment(
            name="multiscale",
            model_class=SimpleVAEWithHyperbolic,
            padic_loss_class=MultiscalePadicLoss,
        )
    )

    # Dual-head architecture
    print("\n--- Dual-Head Architecture ---")

    results.append(
        run_experiment(
            name="dual_head_soft",
            model_class=DualHeadVAE,
            padic_loss_class=SoftPadicRankingLoss,
            use_padic_head=True,
        )
    )

    results.append(
        run_experiment(
            name="dual_head_contrastive",
            model_class=DualHeadVAE,
            padic_loss_class=ContrastivePadicLoss,
            use_padic_head=True,
        )
    )

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Experiment':<25} {'Spearman':>10} {'Accuracy':>10} {'Time':>8}")
    print("-" * 60)

    for r in results:
        print(f"{r.name:<25} {r.spearman:>+10.4f} {r.accuracy:>9.1%} {r.training_time:>7.1f}s")

    # Find best
    print("\n--- Analysis ---")
    best_corr = max(results, key=lambda r: r.spearman)
    best_acc = max(results, key=lambda r: r.accuracy)

    print(f"Best correlation: {best_corr.name} (Spearman={best_corr.spearman:+.4f})")
    print(f"Best accuracy: {best_acc.name} (Accuracy={best_acc.accuracy:.1%})")

    # Compare to baseline
    baseline = next(r for r in results if r.name == "hyperbolic_no_padic")
    print(f"\nImprovements over hyperbolic_no_padic (Spearman={baseline.spearman:+.4f}):")

    for r in results:
        if r.name != "hyperbolic_no_padic":
            delta = r.spearman - baseline.spearman
            status = "better" if delta > 0 else "worse"
            print(f"  {r.name}: {delta:+.4f} ({status})")

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    improvements = [
        (r.name, r.spearman - baseline.spearman)
        for r in results
        if r.name != "hyperbolic_no_padic" and r.accuracy > 0.95
    ]
    improvements.sort(key=lambda x: -x[1])

    if improvements and improvements[0][1] > 0:
        print(f"\nBest alternative: {improvements[0][0]} (Spearman +{improvements[0][1]:.4f})")
    else:
        print("\nNo alternative beats hyperbolic_no_padic while maintaining accuracy.")
        print("Recommendation: Use hyperbolic projection alone (no explicit p-adic loss).")


if __name__ == "__main__":
    main()
