# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Viral Evolution Predictor using p-adic framework.

This module predicts viral escape mutations and evolutionary trajectories
using p-adic distance metrics and immune pressure modeling.

Key Concepts:
    - P-adic codon distances predict mutational accessibility
    - Immune pressure modeling (antibody binding sites, T-cell epitopes)
    - Fitness landscape based on codon usage and protein stability
    - Escape mutation prediction combining all factors

Applications:
    - Vaccine design: anticipate escape variants
    - Therapeutic development: identify resistant mutations
    - Surveillance: predict emerging variants

Research References:
    - RESEARCH_PROPOSALS/Autoimmunity_Codon_Adaptation/proposal.md
    - COMPREHENSIVE_RESEARCH_REPORT.md Section 2.8
"""

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import torch
import torch.nn as nn

from src.geometry import poincare_distance


class SelectionType(Enum):
    """Types of evolutionary selection pressure."""

    POSITIVE = "positive"  # Favors change (immune escape)
    NEGATIVE = "negative"  # Disfavors change (functional constraint)
    NEUTRAL = "neutral"  # No selection pressure
    BALANCING = "balancing"  # Maintains diversity


# Amino acid physicochemical properties for mutation impact assessment
AMINO_ACID_PROPERTIES = {
    "A": {"hydrophobicity": 1.8, "volume": 88.6, "charge": 0, "polarity": 0},
    "R": {"hydrophobicity": -4.5, "volume": 173.4, "charge": 1, "polarity": 1},
    "N": {"hydrophobicity": -3.5, "volume": 114.1, "charge": 0, "polarity": 1},
    "D": {"hydrophobicity": -3.5, "volume": 111.1, "charge": -1, "polarity": 1},
    "C": {"hydrophobicity": 2.5, "volume": 108.5, "charge": 0, "polarity": 0},
    "E": {"hydrophobicity": -3.5, "volume": 138.4, "charge": -1, "polarity": 1},
    "Q": {"hydrophobicity": -3.5, "volume": 143.8, "charge": 0, "polarity": 1},
    "G": {"hydrophobicity": -0.4, "volume": 60.1, "charge": 0, "polarity": 0},
    "H": {"hydrophobicity": -3.2, "volume": 153.2, "charge": 0.5, "polarity": 1},
    "I": {"hydrophobicity": 4.5, "volume": 166.7, "charge": 0, "polarity": 0},
    "L": {"hydrophobicity": 3.8, "volume": 166.7, "charge": 0, "polarity": 0},
    "K": {"hydrophobicity": -3.9, "volume": 168.6, "charge": 1, "polarity": 1},
    "M": {"hydrophobicity": 1.9, "volume": 162.9, "charge": 0, "polarity": 0},
    "F": {"hydrophobicity": 2.8, "volume": 189.9, "charge": 0, "polarity": 0},
    "P": {"hydrophobicity": -1.6, "volume": 112.7, "charge": 0, "polarity": 0},
    "S": {"hydrophobicity": -0.8, "volume": 89.0, "charge": 0, "polarity": 1},
    "T": {"hydrophobicity": -0.7, "volume": 116.1, "charge": 0, "polarity": 1},
    "W": {"hydrophobicity": -0.9, "volume": 227.8, "charge": 0, "polarity": 0},
    "Y": {"hydrophobicity": -1.3, "volume": 193.6, "charge": 0, "polarity": 1},
    "V": {"hydrophobicity": 4.2, "volume": 140.0, "charge": 0, "polarity": 0},
}


@dataclass
class EscapeMutation:
    """Represents a predicted escape mutation."""

    position: int
    original_aa: str
    mutant_aa: str
    escape_score: float  # 0-1, probability of immune escape
    fitness_cost: float  # 0-1, cost to viral fitness
    padic_distance: float  # P-adic distance between codons
    selection_type: SelectionType = SelectionType.POSITIVE


@dataclass
class MutationHotspot:
    """Region with elevated mutation potential."""

    start: int
    end: int
    mutation_rate: float
    dominant_selection: SelectionType
    epitope_overlap: bool = False
    structural_importance: float = 0.0


@dataclass
class EscapePrediction:
    """Complete prediction result for a viral sequence."""

    mutations: list[EscapeMutation]
    hotspots: list[MutationHotspot]
    overall_escape_risk: float
    trajectory_confidence: float
    key_findings: list[str] = field(default_factory=list)


@dataclass
class EvolutionaryPressure:
    """Evolutionary pressure at a specific site."""

    position: int
    dn_ds_ratio: float  # dN/dS ratio (>1 = positive selection)
    immune_pressure: float  # 0-1
    structural_constraint: float  # 0-1
    codon_accessibility: float  # How many mutations are accessible


class ViralEvolutionPredictor(nn.Module):
    """Predict viral evolution and escape mutations using p-adic framework.

    Combines p-adic codon distances with immune pressure modeling to
    predict likely escape mutations and evolutionary trajectories.
    """

    def __init__(
        self,
        p: int = 3,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        n_epitope_types: int = 3,
    ):
        """Initialize evolution predictor.

        Args:
            p: Prime for p-adic metric (3 for ternary/codon)
            latent_dim: Dimension for sequence embeddings
            hidden_dim: Hidden layer dimension
            n_epitope_types: Number of epitope categories (B-cell, CD4, CD8)
        """
        super().__init__()
        self.p = p
        self.latent_dim = latent_dim
        self.n_epitope_types = n_epitope_types

        # Codon embedding for p-adic distance computation
        self.codon_embedding = nn.Embedding(64, latent_dim)

        # Epitope pressure encoder
        self.epitope_encoder = nn.Sequential(
            nn.Linear(n_epitope_types, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Mutation predictor head
        self.mutation_head = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 20),  # 20 amino acids
            nn.Softmax(dim=-1),
        )

        # Escape score predictor
        self.escape_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Fitness predictor
        self.fitness_head = nn.Sequential(
            nn.Linear(latent_dim + 20, hidden_dim),  # codon + AA properties
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def compute_padic_distance(self, codon1: int, codon2: int) -> float:
        """Compute p-adic distance between two codon indices.

        Args:
            codon1: First codon index (0-63)
            codon2: Second codon index (0-63)

        Returns:
            P-adic distance
        """
        if codon1 == codon2:
            return 0.0

        diff = abs(codon1 - codon2)
        if diff == 0:
            return 0.0

        # Find p-adic valuation (highest power of p dividing diff)
        v = 0
        while diff % self.p == 0:
            diff //= self.p
            v += 1

        return float(self.p ** (-v))

    def compute_mutation_accessibility(self, codon_idx: int) -> dict[int, float]:
        """Compute accessibility of all possible mutations from a codon.

        Args:
            codon_idx: Source codon index

        Returns:
            Dictionary mapping target codon indices to accessibility scores
        """
        accessibility = {}

        for target in range(64):
            if target != codon_idx:
                # P-adic distance gives accessibility (lower = more accessible)
                dist = self.compute_padic_distance(codon_idx, target)
                # Invert: high distance = low accessibility
                accessibility[target] = 1.0 / (1.0 + dist * 10)

        return accessibility

    def encode_epitope_pressure(
        self,
        b_cell_epitopes: torch.Tensor,
        cd4_epitopes: torch.Tensor,
        cd8_epitopes: torch.Tensor,
    ) -> torch.Tensor:
        """Encode immune pressure from epitope annotations.

        Args:
            b_cell_epitopes: B-cell epitope scores (B, L)
            cd4_epitopes: CD4+ T-cell epitope scores (B, L)
            cd8_epitopes: CD8+ T-cell epitope scores (B, L)

        Returns:
            Encoded pressure (B, L, latent_dim)
        """
        # Stack epitope types
        pressure = torch.stack([b_cell_epitopes, cd4_epitopes, cd8_epitopes], dim=-1)

        # Encode
        return self.epitope_encoder(pressure)

    def predict_mutation_probabilities(
        self,
        codon_indices: torch.Tensor,
        epitope_pressure: torch.Tensor,
    ) -> torch.Tensor:
        """Predict probability distribution over mutations at each position.

        Args:
            codon_indices: Current codon sequence (B, L)
            epitope_pressure: Encoded immune pressure (B, L, latent_dim)

        Returns:
            Mutation probabilities (B, L, 20)
        """
        # Embed current codons
        codon_emb = self.codon_embedding(codon_indices)  # (B, L, latent_dim)

        # Combine with pressure
        combined = torch.cat([codon_emb, epitope_pressure], dim=-1)

        # Predict mutations
        return self.mutation_head(combined)

    def compute_escape_scores(
        self,
        codon_indices: torch.Tensor,
        epitope_pressure: torch.Tensor,
    ) -> torch.Tensor:
        """Compute escape probability at each position.

        Args:
            codon_indices: Current codon sequence (B, L)
            epitope_pressure: Encoded immune pressure (B, L, latent_dim)

        Returns:
            Escape scores (B, L)
        """
        # Embed codons
        codon_emb = self.codon_embedding(codon_indices)

        # Add pressure contribution
        combined = codon_emb + epitope_pressure

        return self.escape_head(combined).squeeze(-1)

    def compute_fitness_landscape(
        self,
        codon_indices: torch.Tensor,
        aa_properties: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute fitness at each position for all possible mutations.

        Args:
            codon_indices: Current codon sequence (B, L)
            aa_properties: Optional amino acid property tensor

        Returns:
            Fitness scores (B, L)
        """
        # Embed codons
        codon_emb = self.codon_embedding(codon_indices)

        # Default AA properties if not provided
        if aa_properties is None:
            aa_properties = torch.zeros(codon_indices.shape[0], codon_indices.shape[1], 20, device=codon_indices.device)

        combined = torch.cat([codon_emb, aa_properties], dim=-1)

        return self.fitness_head(combined).squeeze(-1)

    def forward(
        self,
        codon_indices: torch.Tensor,
        epitope_annotations: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass for evolution prediction.

        Args:
            codon_indices: Codon sequence (B, L)
            epitope_annotations: Optional dict with b_cell, cd4, cd8 tensors

        Returns:
            Dictionary with all predictions
        """
        B, L = codon_indices.shape
        device = codon_indices.device

        # Default epitope pressure if not provided
        if epitope_annotations is None:
            b_cell = torch.zeros(B, L, device=device)
            cd4 = torch.zeros(B, L, device=device)
            cd8 = torch.zeros(B, L, device=device)
        else:
            b_cell = epitope_annotations.get("b_cell", torch.zeros(B, L, device=device))
            cd4 = epitope_annotations.get("cd4", torch.zeros(B, L, device=device))
            cd8 = epitope_annotations.get("cd8", torch.zeros(B, L, device=device))

        # Encode epitope pressure
        epitope_pressure = self.encode_epitope_pressure(b_cell, cd4, cd8)

        # Predictions
        mutation_probs = self.predict_mutation_probabilities(codon_indices, epitope_pressure)
        escape_scores = self.compute_escape_scores(codon_indices, epitope_pressure)
        fitness_scores = self.compute_fitness_landscape(codon_indices)

        # Combined escape-fitness score
        combined_scores = escape_scores * (1.0 - fitness_scores * 0.5)  # Escape penalized by fitness cost

        return {
            "mutation_probabilities": mutation_probs,
            "escape_scores": escape_scores,
            "fitness_scores": fitness_scores,
            "combined_scores": combined_scores,
            "epitope_pressure": epitope_pressure,
        }

    def identify_hotspots(
        self,
        escape_scores: torch.Tensor,
        threshold: float = 0.5,
        min_length: int = 3,
    ) -> list[MutationHotspot]:
        """Identify mutation hotspot regions from escape scores.

        Args:
            escape_scores: Escape scores per position (L,)
            threshold: Minimum score to be considered hotspot
            min_length: Minimum length of hotspot region

        Returns:
            List of identified hotspots
        """
        if isinstance(escape_scores, torch.Tensor):
            scores = escape_scores.detach().cpu().numpy()
        else:
            scores = np.array(escape_scores)

        hotspots = []
        in_hotspot = False
        start = 0

        for i, score in enumerate(scores):
            if score >= threshold:
                if not in_hotspot:
                    start = i
                    in_hotspot = True
            else:
                if in_hotspot:
                    if i - start >= min_length:
                        hotspots.append(
                            MutationHotspot(
                                start=start,
                                end=i,
                                mutation_rate=float(np.mean(scores[start:i])),
                                dominant_selection=SelectionType.POSITIVE,
                                epitope_overlap=True,
                            )
                        )
                    in_hotspot = False

        # Check if sequence ends in hotspot
        if in_hotspot and len(scores) - start >= min_length:
            hotspots.append(
                MutationHotspot(
                    start=start,
                    end=len(scores),
                    mutation_rate=float(np.mean(scores[start:])),
                    dominant_selection=SelectionType.POSITIVE,
                    epitope_overlap=True,
                )
            )

        return hotspots

    def predict_escape_mutations(
        self,
        codon_indices: torch.Tensor,
        epitope_annotations: dict[str, torch.Tensor] | None = None,
        top_k: int = 10,
    ) -> EscapePrediction:
        """Generate complete escape mutation predictions.

        Args:
            codon_indices: Codon sequence (L,) or (1, L)
            epitope_annotations: Optional epitope annotations
            top_k: Number of top mutations to return

        Returns:
            Complete prediction with mutations and hotspots
        """
        if codon_indices.dim() == 1:
            codon_indices = codon_indices.unsqueeze(0)

        # Forward pass
        results = self.forward(codon_indices, epitope_annotations)

        escape_scores = results["escape_scores"][0]
        mutation_probs = results["mutation_probabilities"][0]
        fitness_scores = results["fitness_scores"][0]

        # Find top escape positions
        top_positions = torch.topk(escape_scores, min(top_k, len(escape_scores)))

        mutations = []
        aa_list = list("ARNDCEQGHILKMFPSTWYV")

        for _idx, (score, pos) in enumerate(zip(top_positions.values, top_positions.indices, strict=False)):
            pos = pos.item()
            original_codon = codon_indices[0, pos].item()

            # Find most likely mutation
            mut_probs = mutation_probs[pos]
            best_mut_idx = torch.argmax(mut_probs).item()

            mutations.append(
                EscapeMutation(
                    position=pos,
                    original_aa=aa_list[int(original_codon) % 20],  # Simplified mapping
                    mutant_aa=aa_list[int(best_mut_idx)],
                    escape_score=score.item(),
                    fitness_cost=1.0 - fitness_scores[pos].item(),
                    padic_distance=0.33,  # Placeholder
                    selection_type=SelectionType.POSITIVE,
                )
            )

        # Identify hotspots
        hotspots = self.identify_hotspots(escape_scores)

        # Overall metrics
        overall_escape = escape_scores.mean().item()
        trajectory_confidence = 1.0 - escape_scores.std().item()

        # Key findings
        findings = []
        if overall_escape > 0.5:
            findings.append("High overall escape risk detected")
        if len(hotspots) > 3:
            findings.append(f"Multiple mutation hotspots identified ({len(hotspots)})")
        if mutations and mutations[0].escape_score > 0.8:
            findings.append(f"Critical escape site at position {mutations[0].position}")

        return EscapePrediction(
            mutations=mutations,
            hotspots=hotspots,
            overall_escape_risk=overall_escape,
            trajectory_confidence=trajectory_confidence,
            key_findings=findings,
        )

    def compute_evolutionary_pressure(
        self,
        codon_indices: torch.Tensor,
        epitope_pressure: torch.Tensor,
    ) -> list[EvolutionaryPressure]:
        """Compute evolutionary pressure at each position.

        Args:
            codon_indices: Codon sequence (L,)
            epitope_pressure: Encoded epitope pressure (L, latent_dim)

        Returns:
            List of evolutionary pressure metrics per position
        """
        pressures = []

        for pos in range(len(codon_indices)):
            codon = int(codon_indices[pos].item())

            # Compute codon accessibility
            accessibility = self.compute_mutation_accessibility(codon)
            mean_accessibility = float(np.mean(list(accessibility.values())))

            # Immune pressure from epitope encoding
            immune = torch.norm(epitope_pressure[pos]).item()

            # Estimate dN/dS (simplified)
            # High immune pressure + high accessibility = positive selection
            dn_ds = 1.0 + (immune * mean_accessibility)

            pressures.append(
                EvolutionaryPressure(
                    position=pos,
                    dn_ds_ratio=dn_ds,
                    immune_pressure=min(immune, 1.0),
                    structural_constraint=0.5,  # Default
                    codon_accessibility=mean_accessibility,
                )
            )

        return pressures


# =============================================================================
# Transmissibility to Radius Mapping
# =============================================================================
# Maps viral transmissibility metrics to hyperbolic radius in Poincare ball.
# Higher transmissibility = closer to origin (lower radius, central position)
# Lower transmissibility = closer to boundary (higher radius, peripheral)
# =============================================================================


@dataclass
class TransmissibilityProfile:
    """Transmissibility characteristics of a viral variant."""

    variant_name: str
    r0_estimate: float  # Basic reproduction number
    serial_interval: float = 5.0  # Days between generations
    infectivity_peak: float = 1.0  # Relative peak infectivity
    immune_evasion: float = 0.0  # Fraction evading prior immunity
    acr_score: float = 0.0  # ACE2 binding affinity (for SARS-CoV-2)
    embedding: torch.Tensor | None = None


@dataclass
class RadiusMapping:
    """Result of transmissibility to radius mapping."""

    radius: float  # Hyperbolic radius in Poincare ball
    angle: torch.Tensor  # Direction on unit sphere (dim-1 sphere)
    embedding: torch.Tensor  # Full embedding in Poincare ball
    confidence: float  # Confidence in mapping
    transmissibility_score: float  # Combined transmissibility metric


class TransmissibilityRadiusMapper:
    """Map transmissibility metrics to hyperbolic radius.

    Key insight from holographic/hyperbolic geometry:
    - The radial coordinate in the Poincare ball represents hierarchy
    - More "dominant" variants (higher transmissibility) sit closer to origin
    - This creates natural clustering of variant families

    The mapping function is:
        radius = f(transmissibility) = 1 - exp(-alpha * transmissibility)

    where alpha controls the mapping steepness.

    For very high transmissibility, radius approaches 0 (origin).
    For very low transmissibility, radius approaches 1 (boundary).

    Mathematical justification:
    In AdS/CFT, the radial coordinate corresponds to energy scale.
    Higher energy (more fit) variants are UV (boundary).
    However, for hierarchical organization, we invert this:
    Dominant variants at center, rare variants at periphery.
    """

    def __init__(
        self,
        max_radius: float = 0.95,
        min_radius: float = 0.1,
        mapping_steepness: float = 0.5,
        reference_r0: float = 3.0,  # Reference R0 for normalization
        curvature: float = 1.0,
    ):
        """Initialize transmissibility mapper.

        Args:
            max_radius: Maximum radius (for lowest transmissibility)
            min_radius: Minimum radius (for highest transmissibility)
            mapping_steepness: Controls how quickly radius changes with R0
            reference_r0: Reference R0 for normalization (typical variant)
            curvature: Poincare ball curvature
        """
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.alpha = mapping_steepness
        self.reference_r0 = reference_r0
        self.curvature = curvature

    def compute_transmissibility_score(
        self,
        profile: TransmissibilityProfile,
    ) -> float:
        """Compute combined transmissibility score from profile.

        Args:
            profile: Transmissibility profile

        Returns:
            Combined transmissibility score (0-1 normalized)
        """
        # Normalize R0 relative to reference
        r0_normalized = profile.r0_estimate / self.reference_r0

        # Combine factors
        score = (
            0.5 * r0_normalized
            + 0.2 * profile.infectivity_peak
            + 0.2 * profile.immune_evasion
            + 0.1 * profile.acr_score
        )

        # Sigmoid to bound between 0 and 1
        return float(1.0 / (1.0 + math.exp(-2 * (score - 1))))

    def transmissibility_to_radius(
        self,
        transmissibility: float,
    ) -> float:
        """Map transmissibility score to hyperbolic radius.

        Higher transmissibility -> lower radius (closer to origin)

        Args:
            transmissibility: Transmissibility score (0-1)

        Returns:
            Radius in Poincare ball
        """
        # Invert: high transmissibility = low radius
        inverted = 1.0 - transmissibility

        # Apply exponential mapping
        # radius = min + (max - min) * (1 - exp(-alpha * inverted))
        radius = self.min_radius + (self.max_radius - self.min_radius) * (1.0 - math.exp(-self.alpha * inverted))

        return radius

    def radius_to_transmissibility(
        self,
        radius: float,
    ) -> float:
        """Inverse mapping: radius to transmissibility.

        Args:
            radius: Radius in Poincare ball

        Returns:
            Estimated transmissibility score
        """
        # Solve for inverted from radius
        # radius = min + (max - min) * (1 - exp(-alpha * inverted))
        # (radius - min) / (max - min) = 1 - exp(-alpha * inverted)
        # exp(-alpha * inverted) = 1 - (radius - min) / (max - min)

        ratio = (radius - self.min_radius) / (self.max_radius - self.min_radius)
        ratio = max(1e-10, min(1 - 1e-10, ratio))  # Clamp for log

        inverted = -math.log(1 - ratio) / self.alpha
        inverted = max(0, min(1, inverted))

        return 1.0 - inverted

    def map_to_embedding(
        self,
        profile: TransmissibilityProfile,
        direction: torch.Tensor | None = None,
        latent_dim: int = 16,
    ) -> RadiusMapping:
        """Map transmissibility profile to full Poincare ball embedding.

        Args:
            profile: Transmissibility profile
            direction: Optional direction vector (defaults to random)
            latent_dim: Dimension of embedding

        Returns:
            RadiusMapping with full embedding
        """
        # Compute transmissibility score
        score = self.compute_transmissibility_score(profile)

        # Map to radius
        radius = self.transmissibility_to_radius(score)

        # Get direction (either provided or from existing embedding)
        if direction is not None:
            angle = direction / (torch.norm(direction) + 1e-10)
        elif profile.embedding is not None:
            # Extract direction from existing embedding
            angle = profile.embedding / (torch.norm(profile.embedding) + 1e-10)
        else:
            # Random direction
            angle = torch.randn(latent_dim)
            angle = angle / torch.norm(angle)

        # Create embedding at specified radius
        embedding = radius * angle

        return RadiusMapping(
            radius=radius,
            angle=angle,
            embedding=embedding,
            confidence=1.0 - abs(score - 0.5),  # More confident near extremes
            transmissibility_score=score,
        )

    def adjust_embedding_radius(
        self,
        embedding: torch.Tensor,
        new_transmissibility: float,
    ) -> torch.Tensor:
        """Adjust an existing embedding to a new transmissibility level.

        Preserves the angular direction while changing the radius.

        Args:
            embedding: Current embedding
            new_transmissibility: Target transmissibility score

        Returns:
            Adjusted embedding
        """
        # Get current direction
        current_norm = torch.norm(embedding, dim=-1, keepdim=True).clamp(min=1e-10)
        direction = embedding / current_norm

        # Compute new radius
        new_radius = self.transmissibility_to_radius(new_transmissibility)

        return new_radius * direction


class EvolutionaryTrajectoryPredictor(nn.Module):
    """Predict evolutionary trajectories using hyperbolic geometry.

    Combines transmissibility mapping with geodesic evolution
    to predict how variants will evolve over time.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        curvature: float = 1.0,
    ):
        """Initialize trajectory predictor.

        Args:
            latent_dim: Dimension of latent embeddings
            hidden_dim: Hidden layer dimension
            curvature: Poincare ball curvature
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.curvature = curvature

        self.mapper = TransmissibilityRadiusMapper(curvature=curvature)

        # Evolution velocity predictor
        self.velocity_net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),  # +1 for radius
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Transmissibility change predictor
        self.transmissibility_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),  # Bounded change
        )

    def predict_trajectory(
        self,
        embedding: torch.Tensor,
        n_steps: int = 10,
        dt: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict evolutionary trajectory from current state.

        Args:
            embedding: Current embedding (batch, dim)
            n_steps: Number of prediction steps
            dt: Time step size

        Returns:
            Tuple of (trajectory embeddings, transmissibility trajectory)
        """
        batch_size = embedding.size(0)
        device = embedding.device

        trajectory = [embedding]
        # V5.12.2: Use hyperbolic distance for radius computation
        origin = torch.zeros_like(embedding)
        init_radii = poincare_distance(embedding, origin, c=self.curvature)
        transmissibility = [
            torch.tensor(
                [self.mapper.radius_to_transmissibility(init_radii[i].item()) for i in range(batch_size)], device=device
            )
        ]

        current = embedding
        for _ in range(n_steps - 1):
            # V5.12.2: Compute hyperbolic radius feature
            origin_current = torch.zeros_like(current)
            radius = poincare_distance(current, origin_current, c=self.curvature).unsqueeze(-1)

            # Predict velocity in tangent space
            input_features = torch.cat([current, radius], dim=-1)
            velocity = self.velocity_net(input_features)

            # Predict transmissibility change
            trans_change = self.transmissibility_net(current).squeeze(-1)

            # Update position (simple Euler integration)
            # In practice, would use Riemannian exponential map
            current = current + dt * velocity

            # Project back to Poincare ball
            norm = torch.norm(current, dim=-1, keepdim=True)
            max_norm = self.mapper.max_radius
            current = torch.where(
                norm > max_norm,
                current * max_norm / norm,
                current,
            )

            # Update transmissibility
            current_trans = transmissibility[-1] + dt * trans_change
            current_trans = torch.clamp(current_trans, 0, 1)

            trajectory.append(current)
            transmissibility.append(current_trans)

        return torch.stack(trajectory, dim=1), torch.stack(transmissibility, dim=1)

    def forward(
        self,
        embedding: torch.Tensor,
        n_steps: int = 10,
    ) -> dict[str, torch.Tensor]:
        """Forward pass predicting trajectory.

        Args:
            embedding: Current variant embeddings
            n_steps: Number of prediction steps

        Returns:
            Dictionary with trajectory predictions
        """
        traj, trans = self.predict_trajectory(embedding, n_steps)

        return {
            "trajectory": traj,
            "transmissibility": trans,
            "final_embedding": traj[:, -1],
            "final_transmissibility": trans[:, -1],
        }
