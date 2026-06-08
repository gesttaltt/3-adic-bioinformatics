# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""CRISPR off-target analysis pipeline.

This module provides the main analysis class that combines p-adic distance,
hyperbolic embedding, and activity prediction for comprehensive off-target
analysis.

Single responsibility: Analysis orchestration.
"""

from typing import Any

import torch

from .embedder import HyperbolicOfftargetEmbedder
from .padic_distance import PAdicSequenceDistance
from .predictor import OfftargetActivityPredictor
from .types import GuideSafetyProfile, OffTargetSite


class CRISPROfftargetAnalyzer:
    """Complete CRISPR off-target analysis pipeline.

    Combines p-adic distance computation, hyperbolic embedding,
    and activity prediction for comprehensive off-target analysis.

    Attributes:
        p: Prime for p-adic calculations
        activity_threshold: Threshold for high-risk off-targets
        padic_distance: PAdicSequenceDistance instance
        embedder: HyperbolicOfftargetEmbedder instance
        activity_predictor: OfftargetActivityPredictor instance
    """

    def __init__(
        self,
        p: int = 3,
        embedding_dim: int = 64,
        activity_threshold: float = 0.1,
    ):
        """Initialize analyzer.

        Args:
            p: Prime for p-adic calculations
            embedding_dim: Dimension for hyperbolic embeddings
            activity_threshold: Threshold for high-risk off-targets
        """
        self.p = p
        self.activity_threshold = activity_threshold

        self.padic_distance = PAdicSequenceDistance(p=p)
        self.embedder = HyperbolicOfftargetEmbedder(embedding_dim=embedding_dim)
        self.activity_predictor = OfftargetActivityPredictor()

    def find_mismatches(
        self,
        target: str,
        offtarget: str,
    ) -> list[tuple[int, str, str]]:
        """Find all mismatches between sequences.

        Args:
            target: Target sequence
            offtarget: Off-target sequence

        Returns:
            List of (position, target_nt, offtarget_nt)

        Raises:
            ValueError: If sequences have different lengths
        """
        target = target.upper()
        offtarget = offtarget.upper()

        # Validate sequence lengths match
        if len(target) != len(offtarget):
            raise ValueError(
                f"Sequence length mismatch: target={len(target)}, "
                f"offtarget={len(offtarget)}. Sequences must be same length."
            )

        mismatches = []
        for i, (t, o) in enumerate(zip(target, offtarget, strict=False)):
            if t != o:
                mismatches.append((i, t, o))
        return mismatches

    def analyze_offtarget(
        self,
        guide: str,
        offtarget_seq: str,
        chromosome: str = "unknown",
        position: int = 0,
        strand: str = "+",
        pam: str = "NGG",
    ) -> OffTargetSite:
        """Analyze a single off-target site.

        Args:
            guide: Guide RNA sequence (20nt)
            offtarget_seq: Off-target DNA sequence
            chromosome: Chromosome name
            position: Genomic position
            strand: Strand (+ or -)
            pam: PAM sequence

        Returns:
            OffTargetSite with analysis results
        """
        # Find mismatches
        mismatches = self.find_mismatches(guide, offtarget_seq)
        seed_mismatches = sum(1 for pos, _, _ in mismatches if pos >= 12)

        # Compute p-adic distance
        padic_dist = self.padic_distance.compute_distance(guide, offtarget_seq)

        # Compute hyperbolic distance
        with torch.no_grad():
            result = self.embedder([guide], [offtarget_seq])
            hyper_dist = result["hyperbolic_distances"][0].item()

        # Predict activity
        with torch.no_grad():
            activity = self.activity_predictor([guide], [offtarget_seq])
            predicted_activity = activity[0].item()

        return OffTargetSite(
            sequence=offtarget_seq,
            chromosome=chromosome,
            position=position,
            strand=strand,
            pam=pam,
            mismatches=mismatches,
            mismatch_count=len(mismatches),
            seed_mismatches=seed_mismatches,
            padic_distance=padic_dist,
            hyperbolic_distance=hyper_dist,
            predicted_activity=predicted_activity,
        )

    def analyze_guide(
        self,
        guide: str,
        offtarget_sequences: list[tuple[str, dict[str, Any]]],
    ) -> GuideSafetyProfile:
        """Analyze safety profile for a guide RNA.

        Args:
            guide: Guide RNA sequence
            offtarget_sequences: List of (sequence, metadata) tuples

        Returns:
            GuideSafetyProfile with safety analysis
        """
        offtargets = []

        for seq, metadata in offtarget_sequences:
            site = self.analyze_offtarget(
                guide,
                seq,
                chromosome=metadata.get("chromosome", "unknown"),
                position=metadata.get("position", 0),
                strand=metadata.get("strand", "+"),
                pam=metadata.get("pam", "NGG"),
            )
            offtargets.append(site)

        # Compute summary statistics
        high_risk = [o for o in offtargets if o.predicted_activity > self.activity_threshold]
        seed_hits = [o for o in offtargets if o.seed_mismatches <= 1]
        min_padic = min((o.padic_distance for o in offtargets), default=float("inf"))

        # Compute safety radius (minimum hyperbolic distance to high-risk site)
        safety_radius = min(o.hyperbolic_distance for o in high_risk) if high_risk else float("inf")

        # Compute specificity score
        if offtargets:
            avg_distance = sum(o.padic_distance for o in offtargets) / len(offtargets)
            high_risk_ratio = len(high_risk) / len(offtargets)
            specificity = (1 - high_risk_ratio) * min(1.0, avg_distance)
        else:
            specificity = 1.0

        return GuideSafetyProfile(
            guide_sequence=guide,
            total_offtargets=len(offtargets),
            high_risk_offtargets=len(high_risk),
            seed_region_offtargets=len(seed_hits),
            min_padic_distance=min_padic,
            safety_radius=safety_radius,
            specificity_score=specificity,
            recommended=specificity > 0.7 and len(high_risk) < 3,
        )

    def compare_guides(
        self,
        guides: list[str],
        offtarget_db: dict[str, list[tuple[str, dict[str, Any]]]],
    ) -> list[GuideSafetyProfile]:
        """Compare safety profiles of multiple guides.

        Args:
            guides: List of guide sequences
            offtarget_db: Dictionary mapping guide -> off-target list

        Returns:
            List of GuideSafetyProfile sorted by specificity
        """
        profiles = []
        for guide in guides:
            offtargets = offtarget_db.get(guide, [])
            profile = self.analyze_guide(guide, offtargets)
            profiles.append(profile)

        # Sort by specificity (highest first)
        profiles.sort(key=lambda p: p.specificity_score, reverse=True)
        return profiles

    def compute_landscape(
        self,
        guide: str,
        offtarget_sequences: list[str],
    ) -> dict[str, torch.Tensor]:
        """Compute hyperbolic landscape of off-targets.

        Args:
            guide: Guide RNA sequence
            offtarget_sequences: List of off-target sequences

        Returns:
            Dictionary with embeddings and distance matrices
        """
        # Get embeddings
        with torch.no_grad():
            target_result = self.embedder([guide])
            target_emb = target_result["target_embeddings"]

            offtarget_result = self.embedder(
                [guide] * len(offtarget_sequences),
                offtarget_sequences,
            )
            offtarget_embs = offtarget_result["offtarget_embeddings"]

        # Compute pairwise distances
        n = len(offtarget_sequences)
        padic_matrix = torch.zeros(n, n)
        for i in range(n):
            for j in range(n):
                padic_matrix[i, j] = self.padic_distance.compute_distance(
                    offtarget_sequences[i], offtarget_sequences[j]
                )

        return {
            "target_embedding": target_emb,
            "offtarget_embeddings": offtarget_embs,
            "padic_distance_matrix": padic_matrix,
            "distances_to_target": offtarget_result["hyperbolic_distances"],
        }


__all__ = ["CRISPROfftargetAnalyzer"]
