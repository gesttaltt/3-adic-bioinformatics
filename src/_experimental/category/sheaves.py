# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Sheaf Theory for Protein Structure Constraints.

Implements sheaf-theoretic constraints for ensuring local-to-global
consistency in protein modeling.

Key Concepts:
1. Presheaf: Assignment of data to open sets with restriction maps
2. Sheaf: Presheaf satisfying locality and gluing axioms
3. Section: Element of F(U) - local data over region U
4. Stalks: Limit of sections at a point - infinitesimal data

Protein Application:
- Open sets = sequence windows / structural domains
- Sections = local properties (hydrophobicity, charge, etc.)
- Restrictions = how properties change under focus
- Gluing = combining domain properties into full protein

The sheaf condition ensures:
- Local predictions are globally consistent
- Domain interactions are properly captured
- Hierarchical structure is respected

Example:
    Sheaf over protein sequence:
    - F(residue) = amino acid properties
    - F(motif) = motif properties
    - F(domain) = domain properties
    - F(protein) = full protein properties

    Restriction: domain properties restrict to motif properties
    Gluing: compatible domain properties give protein properties
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F

T = TypeVar("T")  # Type of sections


@dataclass
class OpenSet:
    """Represents an open set in the protein topology.

    For proteins, open sets correspond to:
    - Individual residues
    - Local windows
    - Secondary structure elements
    - Domains
    - Full protein
    """

    name: str
    start: int  # Start position in sequence
    end: int  # End position
    level: int = 0  # Hierarchy level (0 = residue, higher = domain)
    children: list[OpenSet] = field(default_factory=list)
    parent: OpenSet | None = None

    def __hash__(self):
        return hash((self.name, self.start, self.end))

    def __eq__(self, other):
        if not isinstance(other, OpenSet):
            return False
        return self.name == other.name and self.start == other.start and self.end == other.end

    def contains(self, other: OpenSet) -> bool:
        """Check if this set contains another."""
        return self.start <= other.start and other.end <= self.end

    def intersects(self, other: OpenSet) -> bool:
        """Check if sets overlap."""
        return not (self.end <= other.start or other.end <= self.start)


@dataclass
class ResidueSection:
    """Section of sheaf over a residue or window.

    Contains local data that can be restricted and glued.
    """

    open_set: OpenSet
    data: torch.Tensor  # Local feature vector
    properties: dict[str, float] = field(default_factory=dict)

    def restrict_to(self, smaller_set: OpenSet) -> ResidueSection:
        """Restrict section to smaller open set."""
        if not self.open_set.contains(smaller_set):
            raise ValueError(f"{self.open_set.name} does not contain {smaller_set.name}")

        # Extract relevant portion of data
        relative_start = smaller_set.start - self.open_set.start
        relative_end = smaller_set.end - self.open_set.start

        if self.data.dim() == 1:
            # Single feature vector - project to subset
            restricted_data = self.data.clone()
        else:
            # Sequence of features - slice
            restricted_data = self.data[relative_start:relative_end]

        return ResidueSection(
            open_set=smaller_set,
            data=restricted_data,
            properties={k: v for k, v in self.properties.items()},
        )


@dataclass
class SheafMorphism:
    """Morphism between sheaves (natural transformation of presheaves)."""

    source_sheaf: ProteinSheaf
    target_sheaf: ProteinSheaf
    component_maps: dict[str, nn.Module]  # Maps for each open set


class SheafGluing(nn.Module):
    """Gluing operation for combining local sections into global.

    The gluing axiom states:
    Given sections s_i ∈ F(U_i) that agree on overlaps,
    there exists unique s ∈ F(∪U_i) restricting to each s_i.

    We implement this as a learnable aggregation that:
    1. Checks compatibility on overlaps
    2. Combines sections into global section
    3. Ensures consistency constraints
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int = 4,
        overlap_penalty: float = 0.1,
    ):
        """Initialize gluing module.

        Args:
            hidden_dim: Dimension of sections
            n_heads: Number of attention heads for combination
            overlap_penalty: Penalty for inconsistent overlaps
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.overlap_penalty = overlap_penalty

        # Attention for combining sections
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            batch_first=True,
        )

        # Compatibility check
        self.compatibility_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Final projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def check_compatibility(
        self,
        section1: ResidueSection,
        section2: ResidueSection,
    ) -> tuple[float, torch.Tensor]:
        """Check if two sections are compatible on their overlap.

        Args:
            section1: First section
            section2: Second section

        Returns:
            Tuple of (compatibility_score, overlap_embedding)
        """
        if not section1.open_set.intersects(section2.open_set):
            # No overlap - trivially compatible
            return 1.0, torch.zeros(self.hidden_dim, device=section1.data.device)

        # Find overlap region
        overlap_start = max(section1.open_set.start, section2.open_set.start)
        overlap_end = min(section1.open_set.end, section2.open_set.end)

        overlap_set = OpenSet(
            name=f"overlap_{overlap_start}_{overlap_end}",
            start=overlap_start,
            end=overlap_end,
        )

        # Restrict both sections to overlap
        restricted1 = section1.restrict_to(overlap_set)
        restricted2 = section2.restrict_to(overlap_set)

        # Check compatibility
        combined = torch.cat([restricted1.data, restricted2.data], dim=-1)
        if combined.dim() == 1:
            combined = combined.unsqueeze(0)

        compatibility = self.compatibility_net(combined).mean()

        # Overlap embedding is the average
        overlap_emb = (restricted1.data + restricted2.data) / 2
        if overlap_emb.dim() > 1:
            overlap_emb = overlap_emb.mean(dim=0)

        return compatibility.item(), overlap_emb

    def glue(
        self,
        sections: list[ResidueSection],
    ) -> tuple[torch.Tensor, float]:
        """Glue multiple sections into a global section.

        Args:
            sections: List of local sections

        Returns:
            Tuple of (global_section, consistency_score)
        """
        if not sections:
            raise ValueError("Cannot glue empty list of sections")

        # Stack section data
        section_data = []
        for s in sections:
            if s.data.dim() == 1:
                section_data.append(s.data.unsqueeze(0))
            else:
                section_data.append(s.data.mean(dim=0, keepdim=True))

        section_tensor = torch.cat(section_data, dim=0).unsqueeze(0)  # (1, n_sections, dim)

        # Check pairwise compatibility
        total_compatibility = 0.0
        n_pairs = 0
        for i, s1 in enumerate(sections):
            for j, s2 in enumerate(sections):
                if i < j:
                    compat, _ = self.check_compatibility(s1, s2)
                    total_compatibility += compat
                    n_pairs += 1

        consistency = total_compatibility / max(n_pairs, 1)

        # Combine via attention
        combined, _ = self.attention(section_tensor, section_tensor, section_tensor)
        combined = combined.squeeze(0)

        # Project to output
        global_section = self.output_proj(combined.mean(dim=0))

        return global_section, consistency


class SheafConstraint(nn.Module):
    """Constraint layer enforcing sheaf conditions.

    Ensures that local predictions satisfy:
    1. Locality: Consistent on restrictions
    2. Gluing: Compatible sections can be combined
    3. Hierarchy: Higher levels aggregate lower levels
    """

    def __init__(
        self,
        hidden_dim: int,
        n_levels: int = 3,
        locality_weight: float = 0.1,
        gluing_weight: float = 0.1,
    ):
        """Initialize sheaf constraint.

        Args:
            hidden_dim: Dimension of features
            n_levels: Number of hierarchy levels
            locality_weight: Weight for locality constraint
            gluing_weight: Weight for gluing constraint
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_levels = n_levels
        self.locality_weight = locality_weight
        self.gluing_weight = gluing_weight

        # Restriction maps between levels
        self.restrictions = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_levels - 1)])

        # Gluing modules per level
        self.gluings = nn.ModuleList([SheafGluing(hidden_dim) for _ in range(n_levels - 1)])

    def locality_loss(
        self,
        higher_level: torch.Tensor,
        lower_level: torch.Tensor,
        level_idx: int,
    ) -> torch.Tensor:
        """Compute locality constraint loss.

        Ensures restriction(higher) ≈ lower for overlapping regions.

        Args:
            higher_level: Features at higher level (batch, hidden)
            lower_level: Features at lower level (batch, hidden)
            level_idx: Index of restriction map to use

        Returns:
            Locality loss
        """
        restricted = self.restrictions[level_idx](higher_level)
        return F.mse_loss(restricted, lower_level)

    def gluing_loss(
        self,
        sections: list[torch.Tensor],
        glued: torch.Tensor,
        level_idx: int,
    ) -> torch.Tensor:
        """Compute gluing constraint loss.

        Ensures glue(sections) is consistent.

        Args:
            sections: Local section features
            glued: Pre-computed glued result
            level_idx: Level index

        Returns:
            Gluing consistency loss
        """
        # Create ResidueSection objects
        section_objs = []
        for i, s in enumerate(sections):
            open_set = OpenSet(f"section_{i}", i, i + 1, level=level_idx)
            section_objs.append(ResidueSection(open_set, s))

        # Compute gluing
        computed_glue, consistency = self.gluings[level_idx].glue(section_objs)

        # Loss is difference from expected + consistency penalty
        reconstruction_loss = F.mse_loss(computed_glue, glued)
        consistency_loss = 1.0 - consistency

        return reconstruction_loss + consistency_loss

    def forward(
        self,
        features_by_level: list[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Apply sheaf constraints and compute losses.

        Args:
            features_by_level: List of features at each hierarchy level
                Level 0 = finest (residues), higher = coarser (domains)

        Returns:
            Tuple of (constrained_features, losses_dict)
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=features_by_level[0].device)

        # Locality constraints
        for i in range(len(features_by_level) - 1):
            loc_loss = self.locality_loss(
                features_by_level[i + 1],
                features_by_level[i],
                i,
            )
            losses[f"locality_level_{i}"] = loc_loss.item()
            total_loss = total_loss + self.locality_weight * loc_loss

        # Gluing constraints (level i sections should glue to level i+1)
        for i in range(len(features_by_level) - 1):
            # Create sections from level i
            level_i = features_by_level[i]
            sections = [level_i[j] for j in range(level_i.size(0))] if level_i.dim() == 2 else [level_i]

            glue_loss = self.gluing_loss(
                sections,
                features_by_level[i + 1].mean(dim=0)
                if features_by_level[i + 1].dim() > 1
                else features_by_level[i + 1],
                i,
            )
            losses[f"gluing_level_{i}"] = glue_loss.item()
            total_loss = total_loss + self.gluing_weight * glue_loss

        losses["total_sheaf_loss"] = total_loss.item()

        return total_loss, losses


class ProteinSheaf(nn.Module):
    """Sheaf of features over a protein sequence.

    Assigns feature vectors to each open set (residue, motif, domain, protein)
    with consistent restriction maps and gluing operations.
    """

    def __init__(
        self,
        feature_dim: int,
        n_levels: int = 4,  # residue, motif, domain, protein
        use_attention: bool = True,
    ):
        """Initialize protein sheaf.

        Args:
            feature_dim: Dimension of feature vectors (sections)
            n_levels: Number of hierarchy levels
            use_attention: Use attention for aggregation
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.n_levels = n_levels

        # Feature extractors per level
        self.level_encoders = nn.ModuleList([nn.Linear(feature_dim, feature_dim) for _ in range(n_levels)])

        # Sheaf constraint
        self.constraint = SheafConstraint(feature_dim, n_levels)

        # Gluing for global section
        self.global_gluing = SheafGluing(feature_dim)

    def compute_sections(
        self,
        sequence_features: torch.Tensor,
        open_sets: list[OpenSet],
    ) -> dict[str, ResidueSection]:
        """Compute sections over given open sets.

        Args:
            sequence_features: Features for sequence (seq_len, feature_dim)
            open_sets: List of open sets to compute sections for

        Returns:
            Dictionary mapping open set names to sections
        """
        sections = {}

        for open_set in open_sets:
            # Extract features for this open set
            local_features = sequence_features[open_set.start : open_set.end]

            # Encode at appropriate level
            encoded = self.level_encoders[open_set.level](local_features)

            # Create section
            section = ResidueSection(
                open_set=open_set,
                data=encoded,
            )
            sections[open_set.name] = section

        return sections

    def forward(
        self,
        sequence_features: torch.Tensor,
        hierarchy: list[list[OpenSet]] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Process sequence through protein sheaf.

        Args:
            sequence_features: Input features (batch, seq_len, dim) or (seq_len, dim)
            hierarchy: Optional hierarchy of open sets per level

        Returns:
            Tuple of (global_section, sheaf_losses)
        """
        if sequence_features.dim() == 2:
            sequence_features = sequence_features.unsqueeze(0)

        batch_size, seq_len, _ = sequence_features.shape

        # Build default hierarchy if not provided
        if hierarchy is None:
            hierarchy = self._build_default_hierarchy(seq_len)

        # Process each batch element
        global_sections = []
        all_losses = {f"locality_level_{i}": 0.0 for i in range(self.n_levels - 1)}
        all_losses.update({f"gluing_level_{i}": 0.0 for i in range(self.n_levels - 1)})
        all_losses["total_sheaf_loss"] = 0.0

        for b in range(batch_size):
            features_by_level = []

            for level_idx, level_sets in enumerate(hierarchy):
                level_features = []
                for open_set in level_sets:
                    local = sequence_features[b, open_set.start : open_set.end]
                    encoded = self.level_encoders[level_idx](local)
                    level_features.append(encoded.mean(dim=0))

                features_by_level.append(torch.stack(level_features))

            # Apply sheaf constraints
            loss, losses = self.constraint(features_by_level)

            for k, v in losses.items():
                all_losses[k] += v / batch_size

            # Glue to get global section
            top_sections = [
                ResidueSection(
                    OpenSet(f"top_{i}", 0, seq_len, level=self.n_levels - 1),
                    features_by_level[-1][i],
                )
                for i in range(features_by_level[-1].size(0))
            ]

            global_sec, _ = self.global_gluing.glue(top_sections)
            global_sections.append(global_sec)

        return torch.stack(global_sections), all_losses

    def _build_default_hierarchy(
        self,
        seq_len: int,
    ) -> list[list[OpenSet]]:
        """Build default hierarchy of open sets.

        Args:
            seq_len: Sequence length

        Returns:
            List of lists of open sets per level
        """
        hierarchy = []

        # Level 0: individual residues
        level0 = [OpenSet(f"residue_{i}", i, i + 1, level=0) for i in range(seq_len)]
        hierarchy.append(level0)

        # Level 1: windows of 5
        window_size = 5
        level1 = []
        for i in range(0, seq_len, window_size):
            end = min(i + window_size, seq_len)
            level1.append(OpenSet(f"window_{i}", i, end, level=1))
        hierarchy.append(level1)

        # Level 2: quarters
        quarter = max(1, seq_len // 4)
        level2 = []
        for i in range(0, seq_len, quarter):
            end = min(i + quarter, seq_len)
            level2.append(OpenSet(f"quarter_{i}", i, end, level=2))
        hierarchy.append(level2)

        # Level 3: full protein
        level3 = [OpenSet("protein", 0, seq_len, level=3)]
        hierarchy.append(level3)

        return hierarchy


__all__ = [
    "OpenSet",
    "ResidueSection",
    "SheafMorphism",
    "SheafGluing",
    "SheafConstraint",
    "ProteinSheaf",
]
