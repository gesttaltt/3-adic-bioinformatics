"""Curriculum Training for VAE with staged loss introduction.

Implements a phased training approach:
1. Phase 1: Reconstruction only (warm-up)
2. Phase 2: Add KL divergence (standard VAE)
3. Phase 3: Add p-adic structure losses
4. Phase 4: Full training with all losses

This staged approach helps the model learn good reconstruction before
imposing structural constraints, preventing early collapse.

Usage:
    from src.training.curriculum_trainer import CurriculumTrainer

    trainer = CurriculumTrainer(model, config)
    results = trainer.train(ops, indices)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CurriculumConfig:
    """Configuration for curriculum training."""

    # Phase durations (in epochs)
    phase1_epochs: int = 20  # Reconstruction only
    phase2_epochs: int = 20  # + KL
    phase3_epochs: int = 20  # + P-adic
    phase4_epochs: int = 40  # Full training

    # Learning rates per phase
    phase1_lr: float = 0.01  # Higher LR for reconstruction
    phase2_lr: float = 0.005
    phase3_lr: float = 0.003
    phase4_lr: float = 0.001  # Lower LR for fine-tuning

    # Beta schedule within phases
    beta_max: float = 0.1
    beta_warmup: bool = True

    # Loss weights
    padic_weight: float = 0.5
    radial_weight: float = 0.3

    # Loss types
    padic_loss_type: str = "triplet"
    radial_loss_type: str = "monotonic"

    # Transition smoothness
    smooth_transition_epochs: int = 5

    @property
    def total_epochs(self) -> int:
        return self.phase1_epochs + self.phase2_epochs + self.phase3_epochs + self.phase4_epochs


class CurriculumTrainer:
    """Trainer with curriculum-based loss introduction."""

    def __init__(
        self,
        model: nn.Module,
        config: CurriculumConfig | None = None,
        device: str = "cpu",
    ):
        self.model = model
        self.config = config or CurriculumConfig()
        self.device = device
        self.model.to(device)

        # Initialize losses
        self._init_losses()

        # Training history
        self.history: dict[str, list[float]] = {
            "loss": [],
            "recon": [],
            "kl": [],
            "padic": [],
            "radial": [],
            "accuracy": [],
            "spearman": [],
            "phase": [],
            "epoch": [],
        }

    def _init_losses(self):
        """Initialize loss functions."""
        from src.losses.dual_vae_loss import KLDivergenceLoss, ReconstructionLoss

        self.recon_fn = ReconstructionLoss()
        self.kl_fn = KLDivergenceLoss()

        # P-adic loss
        if self.config.padic_loss_type == "triplet":
            from src.losses.padic import PAdicRankingLoss

            self.padic_fn = PAdicRankingLoss(margin=0.1, n_triplets=500)
        elif self.config.padic_loss_type == "soft_ranking":
            self.padic_fn = SoftPadicRankingLoss(temperature=0.5)
        else:
            self.padic_fn = None

        # Radial loss
        if self.config.radial_loss_type == "monotonic":
            from src.losses.padic_geodesic import MonotonicRadialLoss

            self.radial_fn = MonotonicRadialLoss()
        elif self.config.radial_loss_type == "hierarchy":
            from src.losses.padic_geodesic import RadialHierarchyLoss

            self.radial_fn = RadialHierarchyLoss()
        else:
            self.radial_fn = None

    def get_phase(self, epoch: int) -> tuple[int, float, float, float, float]:
        """Get current phase and loss weights.

        Returns: (phase, recon_weight, kl_weight, padic_weight, radial_weight)
        """
        c = self.config
        p1_end = c.phase1_epochs
        p2_end = p1_end + c.phase2_epochs
        p3_end = p2_end + c.phase3_epochs

        # Smooth transition function
        def smooth_weight(epoch_in_phase: int, transition_epochs: int) -> float:
            if transition_epochs <= 0:
                return 1.0
            return min(1.0, epoch_in_phase / transition_epochs)

        if epoch < p1_end:
            # Phase 1: Reconstruction only
            return 1, 1.0, 0.0, 0.0, 0.0

        elif epoch < p2_end:
            # Phase 2: Add KL
            epoch_in_phase = epoch - p1_end
            kl_weight = smooth_weight(epoch_in_phase, c.smooth_transition_epochs)

            # Beta warmup within phase
            beta = c.beta_max * min(1.0, (epoch_in_phase + 1) / 10) if c.beta_warmup else c.beta_max

            return 2, 1.0, beta * kl_weight, 0.0, 0.0

        elif epoch < p3_end:
            # Phase 3: Add p-adic
            epoch_in_phase = epoch - p2_end
            padic_weight = c.padic_weight * smooth_weight(epoch_in_phase, c.smooth_transition_epochs)
            radial_weight = c.radial_weight * smooth_weight(epoch_in_phase, c.smooth_transition_epochs)

            # Cyclical beta
            beta = c.beta_max * (0.5 + 0.5 * np.sin(2 * np.pi * (epoch - p1_end) / 50))

            return 3, 1.0, beta, padic_weight, radial_weight

        else:
            # Phase 4: Full training
            epoch_in_phase = epoch - p3_end

            # Cyclical beta
            beta = c.beta_max * (0.5 + 0.5 * np.sin(2 * np.pi * (epoch - p1_end) / 50))

            return 4, 1.0, beta, c.padic_weight, c.radial_weight

    def get_lr(self, phase: int) -> float:
        """Get learning rate for phase."""
        c = self.config
        if phase == 1:
            return c.phase1_lr
        elif phase == 2:
            return c.phase2_lr
        elif phase == 3:
            return c.phase3_lr
        else:
            return c.phase4_lr

    def train_epoch(
        self,
        ops: torch.Tensor,
        indices: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        phase, recon_w, kl_w, padic_w, radial_w = self.get_phase(epoch)

        # Forward pass
        outputs = self.model(ops)
        logits = outputs["logits"]
        mu = outputs["mu"]
        logvar = outputs["logvar"]
        z = outputs.get("z_euc", outputs["z"])

        # Compute losses
        recon = self.recon_fn(logits, ops)
        kl = self.kl_fn(mu, logvar)

        loss = recon_w * recon + kl_w * kl

        # P-adic loss (only in phase 3+)
        padic_val = 0.0
        if self.padic_fn is not None and padic_w > 0:
            padic = self.padic_fn(z, indices)
            if isinstance(padic, tuple):
                padic = padic[0]
            padic_val = padic.item()
            loss = loss + padic_w * padic

        # Radial loss (only in phase 3+)
        radial_val = 0.0
        if self.radial_fn is not None and radial_w > 0:
            radial = self.radial_fn(z, indices)
            if isinstance(radial, tuple):
                radial = radial[0]
            radial_val = radial.item()
            loss = loss + radial_w * radial

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()

        # Compute accuracy
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            target = (ops + 1).long()
            accuracy = (pred == target).float().mean().item()

        return {
            "loss": loss.item(),
            "recon": recon.item(),
            "kl": kl.item(),
            "padic": padic_val,
            "radial": radial_val,
            "accuracy": accuracy,
            "phase": phase,
        }

    def train(
        self,
        ops: torch.Tensor,
        indices: torch.Tensor,
        eval_fn: Callable | None = None,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Full training with curriculum.

        Args:
            ops: Training data (batch, input_dim)
            indices: Operation indices for p-adic distances
            eval_fn: Optional evaluation function (z, indices) -> dict
            verbose: Print progress

        Returns:
            Dict with final metrics and history
        """
        ops = ops.to(self.device)
        indices = indices.to(self.device)

        optimizer = None
        current_phase = 0

        for epoch in range(self.config.total_epochs):
            phase, *_ = self.get_phase(epoch)

            # Update optimizer when phase changes
            if phase != current_phase:
                lr = self.get_lr(phase)
                optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
                current_phase = phase
                if verbose:
                    print(f"\n--- Phase {phase} (lr={lr}) ---")

            if optimizer is None:
                optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.phase1_lr)

            # Train epoch
            metrics = self.train_epoch(ops, indices, optimizer, epoch)

            # Evaluate periodically
            if epoch % 10 == 0:
                if eval_fn is not None:
                    with torch.no_grad():
                        outputs = self.model(ops)
                        z = outputs.get("z_euc", outputs["z"])
                        eval_metrics = eval_fn(z, indices)
                        metrics["spearman"] = eval_metrics.get("spearman", 0.0)
                else:
                    metrics["spearman"] = 0.0

                # Record history
                for key in ["loss", "recon", "kl", "padic", "radial", "accuracy", "spearman", "phase"]:
                    self.history[key].append(metrics.get(key, 0.0))
                self.history["epoch"].append(epoch)

                if verbose:
                    print(
                        f"Epoch {epoch:3d} | Phase {metrics['phase']} | "
                        f"Loss {metrics['loss']:.4f} | Acc {metrics['accuracy']:.1%} | "
                        f"Spearman {metrics.get('spearman', 0.0):+.4f}"
                    )

        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(ops)
            z = outputs.get("z_euc", outputs["z"])
            outputs.get("z_hyp", z)

            pred = outputs["logits"].argmax(dim=-1)
            target = (ops + 1).long()
            final_accuracy = (pred == target).float().mean().item()

            final_metrics = eval_fn(z, indices) if eval_fn is not None else {}

        return {
            "accuracy": final_accuracy,
            "spearman": final_metrics.get("spearman", 0.0),
            "silhouette": final_metrics.get("silhouette", 0.0),
            "history": self.history,
        }


class SoftPadicRankingLoss(nn.Module):
    """Soft p-adic ranking using KL divergence."""

    def __init__(self, temperature: float = 0.5, n_samples: int = 200):
        super().__init__()
        self.temperature = temperature
        self.n_samples = n_samples

    def compute_padic_distance(self, i: int, j: int) -> float:
        if i == j:
            return 0.0
        diff = abs(i - j)
        k = 0
        while diff % 3 == 0:
            diff //= 3
            k += 1
        return 3.0 ** (-k)

    def forward(self, z: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        n = z.size(0)
        if n < 3:
            return torch.tensor(0.0, device=z.device)

        if n > self.n_samples:
            idx = torch.randperm(n)[: self.n_samples]
            z = z[idx]
            indices = indices[idx]
            n = self.n_samples

        latent_dist = torch.cdist(z, z)
        padic_dist = torch.zeros(n, n, device=z.device)
        for i in range(n):
            for j in range(n):
                padic_dist[i, j] = self.compute_padic_distance(indices[i].item(), indices[j].item())

        latent_ranks = F.softmax(-latent_dist / self.temperature, dim=1)
        padic_ranks = F.softmax(-padic_dist / self.temperature, dim=1)

        return F.kl_div(latent_ranks.log(), padic_ranks, reduction="batchmean")


__all__ = [
    "CurriculumTrainer",
    "CurriculumConfig",
    "SoftPadicRankingLoss",
]
