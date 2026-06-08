# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Rich Hierarchy Loss - Hierarchy enforcement with richness preservation.

This module implements a novel approach to radial hierarchy training that
preserves within-level variance (geometric richness) while enforcing
correct 3-adic radial ordering.

Key innovation: Unlike RadialStratificationLoss and RadialHierarchyLoss which
push individual samples to target radii (collapsing variance), RichHierarchyLoss
operates on per-level MEANS, allowing variance within each valuation level.

This approach achieved the best hierarchy-richness balance in the
homeostatic_rich checkpoint (-0.8321 hierarchy with 5.8x more richness).

Usage:
    from src.losses import RichHierarchyLoss

    loss_fn = RichHierarchyLoss(
        hierarchy_weight=5.0,
        coverage_weight=1.0,
        richness_weight=2.0,
        separation_weight=3.0,
    )

    # Requires original_radii for richness comparison
    losses = loss_fn(z_hyp, indices, logits, targets, original_radii)
    total_loss = losses['total']
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core import TERNARY
from src.geometry.poincare import poincare_distance


class RichHierarchyLoss(nn.Module):
    """Loss function balancing hierarchy, coverage, and richness.

    Unlike previous approaches that collapsed richness to maximize hierarchy,
    this loss actively preserves within-level variance while pushing toward
    better hierarchy.

    Components:
        1. Hierarchy: Push mean radius per level toward target (not individual samples)
        2. Coverage: Reconstruction accuracy (cross-entropy)
        3. Richness: Penalize variance collapse below threshold
        4. Separation: Ensure valuation levels don't overlap

    Args:
        inner_radius: Target radius for highest valuation v=9 (near origin)
        outer_radius: Target radius for lowest valuation v=0 (near boundary)
        hierarchy_weight: Weight for hierarchy loss component
        coverage_weight: Weight for coverage (reconstruction) loss
        richness_weight: Weight for richness preservation loss
        separation_weight: Weight for level separation loss
        min_richness_ratio: Minimum ratio of new/original variance before penalty

    Example:
        >>> loss_fn = RichHierarchyLoss(hierarchy_weight=5.0, richness_weight=2.0)
        >>> out = model(batch_ops)
        >>> losses = loss_fn(
        ...     out['z_A_hyp'], batch_idx, logits, batch_ops, original_radii
        ... )
        >>> losses['total'].backward()
    """

    def __init__(
        self,
        inner_radius: float = 0.1,
        outer_radius: float = 0.9,
        hierarchy_weight: float = 5.0,
        coverage_weight: float = 1.0,
        richness_weight: float = 2.0,
        separation_weight: float = 3.0,
        min_richness_ratio: float = 0.5,
        curvature: float = 1.0,
    ):
        super().__init__()
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.hierarchy_weight = hierarchy_weight
        self.coverage_weight = coverage_weight
        self.richness_weight = richness_weight
        self.separation_weight = separation_weight
        self.min_richness_ratio = min_richness_ratio
        self.max_valuation = 9
        self.curvature = curvature

        # Precompute target radii for each valuation level (in hyperbolic distance)
        # v=0 → outer_radius, v=9 → inner_radius
        target_radii = torch.tensor(
            [outer_radius - (v / self.max_valuation) * (outer_radius - inner_radius) for v in range(10)]
        )
        self.register_buffer("target_radii", target_radii)

    def forward(
        self,
        z_hyp: torch.Tensor,
        indices: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
        original_radii: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute combined hierarchy, coverage, richness, and separation losses.

        Args:
            z_hyp: Points in Poincare ball (batch, latent_dim)
            indices: Operation indices (batch,) in range [0, 19682]
            logits: Decoder output logits (batch, 9, 3) or (batch, 27)
            targets: Target operations (batch, 9) with values in {-1, 0, 1}
            original_radii: Original radii before training for richness comparison
                           Shape: (batch,) or (19683,) indexed by indices

        Returns:
            Dict with keys: 'total', 'hierarchy_loss', 'coverage_loss',
                           'richness_loss', 'separation_loss'
        """
        device = z_hyp.device
        # V5.12.2: Use hyperbolic distance instead of Euclidean norm
        # This ensures loss operates in consistent geometry with decoder
        origin = torch.zeros_like(z_hyp)
        radii = poincare_distance(z_hyp, origin, c=self.curvature)
        valuations = TERNARY.valuation(indices).long().to(device)

        # === 1. Hierarchy: Push mean radius per level toward target ===
        # Key innovation: operate on MEANS, not individual samples
        # This allows variance within each level (preserves richness)
        hierarchy_loss = torch.tensor(0.0, device=device)
        unique_vals = torch.unique(valuations)

        for v in unique_vals:
            mask = valuations == v
            if mask.sum() > 0:
                mean_r = radii[mask].mean()
                target_r = self.target_radii[v]
                hierarchy_loss = hierarchy_loss + (mean_r - target_r) ** 2

        hierarchy_loss = hierarchy_loss / len(unique_vals)

        # === 2. Coverage: Reconstruction accuracy ===
        coverage_loss = F.cross_entropy(
            logits.view(-1, 3),
            (targets + 1).long().view(-1),
        )

        # === 3. Richness: Preserve within-level variance ===
        richness_loss = torch.tensor(0.0, device=device)

        if original_radii is not None:
            # Handle both indexed and pre-indexed original_radii
            orig_radii_batch = original_radii[indices] if original_radii.shape[0] == 19683 else original_radii

            for v in unique_vals:
                mask = valuations == v
                if mask.sum() > 1:
                    new_var = radii[mask].var()
                    orig_var = orig_radii_batch[mask].var() + 1e-8

                    # Penalize if variance drops below threshold
                    ratio = new_var / orig_var
                    if ratio < self.min_richness_ratio:
                        # Strong penalty for collapsing variance
                        richness_loss = richness_loss + (self.min_richness_ratio - ratio) ** 2

            richness_loss = richness_loss / max(len(unique_vals), 1)

        # === 4. Separation: Ensure levels don't overlap ===
        separation_loss = torch.tensor(0.0, device=device)
        mean_radii_list = []

        for v in sorted(unique_vals.tolist()):
            mask = valuations == v
            if mask.sum() > 0:
                mean_radii_list.append((v, radii[mask].mean()))

        for i in range(len(mean_radii_list) - 1):
            v1, r1 = mean_radii_list[i]
            v2, r2 = mean_radii_list[i + 1]
            # v2 > v1, so r2 should be < r1 (higher valuation = smaller radius)
            margin = 0.01
            violation = F.relu(r2 - r1 + margin)
            separation_loss = separation_loss + violation

        # === Combine with weights ===
        total = (
            self.hierarchy_weight * hierarchy_loss
            + self.coverage_weight * coverage_loss
            + self.richness_weight * richness_loss
            + self.separation_weight * separation_loss
        )

        return {
            "total": total,
            "hierarchy_loss": hierarchy_loss,
            "coverage_loss": coverage_loss,
            "richness_loss": richness_loss,
            "separation_loss": separation_loss,
        }

    def extra_repr(self) -> str:
        return (
            f"inner_radius={self.inner_radius}, "
            f"outer_radius={self.outer_radius}, "
            f"hierarchy_weight={self.hierarchy_weight}, "
            f"coverage_weight={self.coverage_weight}, "
            f"richness_weight={self.richness_weight}, "
            f"separation_weight={self.separation_weight}, "
            f"min_richness_ratio={self.min_richness_ratio}, "
            f"curvature={self.curvature}"
        )
