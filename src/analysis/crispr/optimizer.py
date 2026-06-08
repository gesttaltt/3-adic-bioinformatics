# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Guide RNA design optimization for CRISPR analysis.

This module provides optimization strategies for guide RNA design
to minimize off-target effects.

Single responsibility: Guide optimization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .analyzer import CRISPROfftargetAnalyzer
    from .types import OffTargetSite


class GuideDesignOptimizer:
    """Optimizes guide RNA design for minimal off-target effects.

    Uses p-adic and hyperbolic analysis to suggest modifications
    that improve specificity.

    Attributes:
        analyzer: CRISPROfftargetAnalyzer instance for analysis
    """

    def __init__(self, analyzer: CRISPROfftargetAnalyzer | None = None):
        """Initialize optimizer.

        Args:
            analyzer: CRISPROfftargetAnalyzer instance. If None, creates a new one.
        """
        if analyzer is None:
            from .analyzer import CRISPROfftargetAnalyzer

            analyzer = CRISPROfftargetAnalyzer()
        self.analyzer = analyzer

    def suggest_modifications(
        self,
        guide: str,
        problematic_offtargets: list[OffTargetSite],
    ) -> list[tuple[int, str, str, float]]:
        """Suggest nucleotide modifications to reduce off-target binding.

        Analyzes problematic off-target sites to identify positions where
        modifications could improve specificity.

        Args:
            guide: Original guide sequence
            problematic_offtargets: High-risk off-target sites

        Returns:
            List of (position, original_nt, suggested_nt, improvement_score)
            sorted by improvement score descending
        """
        suggestions = []

        # Analyze which positions are shared across problematic off-targets
        position_counts: dict[int, int] = {}
        for site in problematic_offtargets:
            for pos, _, _ in site.mismatches:
                position_counts[pos] = position_counts.get(pos, 0) + 1

        # Find positions that could be modified
        for pos in range(len(guide)):
            original_nt = guide[pos]

            # Skip positions that are already mismatched with most off-targets
            if position_counts.get(pos, 0) > len(problematic_offtargets) // 2:
                continue

            # Try alternative nucleotides
            for alt_nt in "ACGT":
                if alt_nt == original_nt:
                    continue

                # Compute improvement score
                # (simplified - would need full re-analysis in practice)
                seed_bonus = 2.0 if pos >= 12 else 1.0
                mismatch_penalty = position_counts.get(pos, 0) / len(problematic_offtargets)
                improvement = seed_bonus * (1 - mismatch_penalty)

                if improvement > 0.5:
                    suggestions.append((pos, original_nt, alt_nt, improvement))

        # Sort by improvement score
        suggestions.sort(key=lambda x: x[3], reverse=True)
        return suggestions[:5]  # Return top 5 suggestions

    def generate_variants(
        self,
        guide: str,
        n_variants: int = 10,
    ) -> list[str]:
        """Generate variant guides with potential improved specificity.

        Creates single-nucleotide variants of the original guide for
        specificity screening.

        Args:
            guide: Original guide sequence
            n_variants: Number of variants to generate

        Returns:
            List of variant guide sequences including the original
        """
        variants = [guide]
        guide_list = list(guide)

        # Generate single-nucleotide variants
        for pos in range(len(guide)):
            for alt_nt in "ACGT":
                if alt_nt != guide[pos]:
                    variant = guide_list.copy()
                    variant[pos] = alt_nt
                    variants.append("".join(variant))

                    if len(variants) >= n_variants:
                        return variants

        return variants[:n_variants]


__all__ = ["GuideDesignOptimizer"]
