# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Peptide Grammar for Biological Sequence Analysis.

Implements context-free and context-sensitive grammars for peptide
and protein sequences, enabling structured analysis of biological motifs.

Grammar Levels for Proteins:
1. Phonology (character level): Individual amino acids
2. Morphology (motif level): Short functional patterns (NxS/T, RGD, etc.)
3. Syntax (domain level): Secondary structure, domain organization
4. Semantics (function level): Binding, catalysis, signaling

Key Motif Types:
- N-glycosylation: N-[^P]-[ST] (Asn-X-Ser/Thr, X != Pro)
- Phosphorylation: [ST]-X-[RK] (Ser/Thr kinase motifs)
- Signal peptides: M-[^P]{20,30}-[LIVATF]-X-[LIVATF]
- PEST sequences: Rich in P, E, S, T (degradation signals)

Mathematical Framework:
We use a production grammar G = (V, T, P, S) where:
- V: Non-terminals (motif types)
- T: Terminals (amino acids)
- P: Production rules
- S: Start symbol

Context-sensitivity is captured through features/attributes on rules.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import torch


class MotifType(Enum):
    """Types of protein motifs."""

    GLYCOSYLATION = "glycosylation"  # N-linked or O-linked
    PHOSPHORYLATION = "phosphorylation"  # Kinase targets
    SIGNAL_PEPTIDE = "signal_peptide"  # Secretion signals
    TRANSMEMBRANE = "transmembrane"  # Membrane-spanning
    BINDING = "binding"  # Receptor/ligand binding
    CATALYTIC = "catalytic"  # Enzyme active sites
    STRUCTURAL = "structural"  # Secondary structure
    UNKNOWN = "unknown"


@dataclass
class ProteinMotif:
    """A detected protein motif."""

    type: MotifType
    pattern: str  # Regex or grammar pattern
    start: int  # Start position in sequence
    end: int  # End position
    sequence: str  # Matched sequence
    confidence: float = 1.0
    attributes: dict[str, any] = field(default_factory=dict)


@dataclass
class GrammarRule:
    """A production rule in the peptide grammar."""

    name: str
    lhs: str  # Left-hand side (non-terminal)
    rhs: list[str]  # Right-hand side (terminals/non-terminals)
    pattern: str | None = None  # Regex pattern
    constraint: Callable[[str], bool] | None = None  # Context constraint
    priority: int = 0  # Higher = applied first
    motif_type: MotifType = MotifType.UNKNOWN


@dataclass
class ParseTree:
    """Parse tree node for a peptide sequence."""

    symbol: str  # Non-terminal or terminal
    children: list[ParseTree] = field(default_factory=list)
    span: tuple[int, int] = (0, 0)  # (start, end) in original sequence
    sequence: str = ""
    rule: GrammarRule | None = None
    embedding: torch.Tensor | None = None

    def is_leaf(self) -> bool:
        """Check if this is a leaf (terminal) node."""
        return len(self.children) == 0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "span": self.span,
            "sequence": self.sequence,
            "children": [c.to_dict() for c in self.children],
        }


class PeptideGrammar:
    """Context-free grammar for peptide sequence analysis.

    Defines production rules for common protein motifs and
    provides parsing capabilities.
    """

    # Standard amino acid alphabet
    AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

    # Amino acid classes for pattern matching
    AA_CLASSES = {
        "hydrophobic": set("AILMFVWY"),
        "polar": set("STNQ"),
        "charged": set("DEKRH"),
        "positive": set("KRH"),
        "negative": set("DE"),
        "small": set("AGSTP"),
        "aromatic": set("FYW"),
        "aliphatic": set("AILV"),
    }

    def __init__(self):
        """Initialize peptide grammar with standard rules."""
        self.rules: list[GrammarRule] = []
        self.non_terminals: set[str] = set()
        self.terminals: set[str] = self.AMINO_ACIDS.copy()

        self._add_standard_rules()

    def _add_standard_rules(self):
        """Add standard protein motif rules."""
        # N-glycosylation motif: N-[^P]-[ST]
        self.add_rule(
            GrammarRule(
                name="n_glycosylation",
                lhs="GLYCO",
                rhs=["N", "X", "ST"],
                pattern=r"N[^P][ST]",
                motif_type=MotifType.GLYCOSYLATION,
                priority=10,
            )
        )

        # Phosphorylation motifs
        self.add_rule(
            GrammarRule(
                name="pka_site",
                lhs="PHOSPHO",
                rhs=["RR", "X", "S"],
                pattern=r"RR.S",
                motif_type=MotifType.PHOSPHORYLATION,
                priority=10,
            )
        )

        self.add_rule(
            GrammarRule(
                name="ck2_site",
                lhs="PHOSPHO",
                rhs=["S", "X", "X", "E"],
                pattern=r"S..E",
                motif_type=MotifType.PHOSPHORYLATION,
                priority=8,
            )
        )

        # RGD cell adhesion motif
        self.add_rule(
            GrammarRule(
                name="rgd_motif",
                lhs="BINDING",
                rhs=["R", "G", "D"],
                pattern=r"RGD",
                motif_type=MotifType.BINDING,
                priority=10,
            )
        )

        # PEST sequence (degradation signal)
        self.add_rule(
            GrammarRule(
                name="pest_sequence",
                lhs="PEST",
                rhs=["PEST_REGION"],
                pattern=r"[PEST]{4,}",
                constraint=lambda s: sum(c in "PEST" for c in s) / len(s) > 0.5,
                motif_type=MotifType.STRUCTURAL,
                priority=5,
            )
        )

        # Nuclear localization signal
        self.add_rule(
            GrammarRule(
                name="nls_monopartite",
                lhs="NLS",
                rhs=["K", "RK", "X", "RK"],
                pattern=r"K[RK].[RK]",
                motif_type=MotifType.SIGNAL_PEPTIDE,
                priority=8,
            )
        )

        # Transmembrane region (hydrophobic stretch)
        self.add_rule(
            GrammarRule(
                name="transmembrane",
                lhs="TM",
                rhs=["HYDRO_REGION"],
                pattern=r"[AILMFVWY]{15,25}",
                motif_type=MotifType.TRANSMEMBRANE,
                priority=5,
            )
        )

        # Zinc finger motif
        self.add_rule(
            GrammarRule(
                name="zinc_finger",
                lhs="ZINC",
                rhs=["C", "X2", "C", "X15", "H", "X3", "H"],
                pattern=r"C.{2}C.{15}H.{3}H",
                motif_type=MotifType.CATALYTIC,
                priority=7,
            )
        )

    def add_rule(self, rule: GrammarRule):
        """Add a production rule to the grammar.

        Args:
            rule: Grammar rule to add
        """
        self.rules.append(rule)
        self.non_terminals.add(rule.lhs)
        self.rules.sort(key=lambda r: -r.priority)  # Sort by priority

    def parse(
        self,
        sequence: str,
        max_depth: int = 10,
    ) -> list[ParseTree]:
        """Parse a peptide sequence using the grammar.

        Args:
            sequence: Amino acid sequence
            max_depth: Maximum parse tree depth

        Returns:
            List of parse trees for detected motifs
        """
        sequence = sequence.upper()
        trees = []

        for rule in self.rules:
            if rule.pattern:
                # Use regex matching
                for match in re.finditer(rule.pattern, sequence):
                    start, end = match.span()
                    matched_seq = match.group()

                    # Check constraint if present
                    if rule.constraint and not rule.constraint(matched_seq):
                        continue

                    # Create parse tree
                    tree = ParseTree(
                        symbol=rule.lhs,
                        span=(start, end),
                        sequence=matched_seq,
                        rule=rule,
                    )

                    # Add children for each character
                    for i, aa in enumerate(matched_seq):
                        tree.children.append(
                            ParseTree(
                                symbol=aa,
                                span=(start + i, start + i + 1),
                                sequence=aa,
                            )
                        )

                    trees.append(tree)

        return trees

    def get_motifs(self, sequence: str) -> list[ProteinMotif]:
        """Extract all motifs from a sequence.

        Args:
            sequence: Amino acid sequence

        Returns:
            List of detected motifs
        """
        trees = self.parse(sequence)
        motifs = []

        for tree in trees:
            if tree.rule:
                motif = ProteinMotif(
                    type=tree.rule.motif_type,
                    pattern=tree.rule.pattern or "",
                    start=tree.span[0],
                    end=tree.span[1],
                    sequence=tree.sequence,
                    confidence=1.0,
                    attributes={"rule_name": tree.rule.name},
                )
                motifs.append(motif)

        return motifs


class MotifParser:
    """Advanced parser for complex protein motifs.

    Supports context-sensitive parsing with lookahead/lookbehind
    and compositional motif detection.
    """

    def __init__(self, grammar: PeptideGrammar | None = None):
        """Initialize motif parser.

        Args:
            grammar: Peptide grammar to use (creates default if None)
        """
        self.grammar = grammar or PeptideGrammar()
        self.motif_cache: dict[str, list[ProteinMotif]] = {}

    def parse_with_context(
        self,
        sequence: str,
        context_window: int = 10,
    ) -> list[ProteinMotif]:
        """Parse sequence considering local context.

        Args:
            sequence: Amino acid sequence
            context_window: Window size for context analysis

        Returns:
            List of motifs with context-adjusted confidence
        """
        base_motifs = self.grammar.get_motifs(sequence)

        # Adjust confidence based on context
        adjusted_motifs = []
        for motif in base_motifs:
            # Get context
            ctx_start = max(0, motif.start - context_window)
            ctx_end = min(len(sequence), motif.end + context_window)

            left_context = sequence[ctx_start : motif.start]
            right_context = sequence[motif.end : ctx_end]

            # Adjust confidence based on context features
            confidence = motif.confidence
            confidence *= self._context_score(left_context, right_context, motif)

            adjusted_motif = ProteinMotif(
                type=motif.type,
                pattern=motif.pattern,
                start=motif.start,
                end=motif.end,
                sequence=motif.sequence,
                confidence=confidence,
                attributes={
                    **motif.attributes,
                    "left_context": left_context,
                    "right_context": right_context,
                },
            )
            adjusted_motifs.append(adjusted_motif)

        return adjusted_motifs

    def _context_score(
        self,
        left: str,
        right: str,
        motif: ProteinMotif,
    ) -> float:
        """Score motif based on surrounding context.

        Args:
            left: Left context
            right: Right context
            motif: The motif being scored

        Returns:
            Context score multiplier (0-1)
        """
        score = 1.0

        # Glycosylation sites prefer accessible regions
        if motif.type == MotifType.GLYCOSYLATION:
            # Check for nearby prolines (disruptive)
            if "PP" in left[-5:] or "PP" in right[:5]:
                score *= 0.5

        # Phosphorylation sites prefer acidic environment
        elif motif.type == MotifType.PHOSPHORYLATION:
            acidic_count = (left + right).count("D") + (left + right).count("E")
            score *= min(1.0, 0.5 + 0.1 * acidic_count)

        return score

    def build_hierarchical_parse(
        self,
        sequence: str,
    ) -> ParseTree:
        """Build hierarchical parse tree for entire sequence.

        Creates a tree structure where:
        - Root represents the full protein
        - Internal nodes represent domains/motifs
        - Leaves represent individual amino acids

        Args:
            sequence: Amino acid sequence

        Returns:
            Root of hierarchical parse tree
        """
        # Create root
        root = ParseTree(
            symbol="PROTEIN",
            span=(0, len(sequence)),
            sequence=sequence,
        )

        # Get all motifs
        motifs = self.parse_with_context(sequence)

        # Sort by position
        motifs.sort(key=lambda m: m.start)

        # Build tree structure
        current_pos = 0
        for motif in motifs:
            # Add intervening sequence as "REGION" node
            if motif.start > current_pos:
                region = ParseTree(
                    symbol="REGION",
                    span=(current_pos, motif.start),
                    sequence=sequence[current_pos : motif.start],
                )
                # Add individual amino acids as leaves
                for i, aa in enumerate(region.sequence):
                    region.children.append(
                        ParseTree(
                            symbol=aa,
                            span=(current_pos + i, current_pos + i + 1),
                            sequence=aa,
                        )
                    )
                root.children.append(region)

            # Add motif node
            motif_node = ParseTree(
                symbol=motif.type.value.upper(),
                span=(motif.start, motif.end),
                sequence=motif.sequence,
            )
            for i, aa in enumerate(motif.sequence):
                motif_node.children.append(
                    ParseTree(
                        symbol=aa,
                        span=(motif.start + i, motif.start + i + 1),
                        sequence=aa,
                    )
                )
            root.children.append(motif_node)

            current_pos = motif.end

        # Add remaining sequence
        if current_pos < len(sequence):
            region = ParseTree(
                symbol="REGION",
                span=(current_pos, len(sequence)),
                sequence=sequence[current_pos:],
            )
            for i, aa in enumerate(region.sequence):
                region.children.append(
                    ParseTree(
                        symbol=aa,
                        span=(current_pos + i, current_pos + i + 1),
                        sequence=aa,
                    )
                )
            root.children.append(region)

        return root


def extract_secondary_structure_grammar(
    sequence: str,
    structure: str,
) -> list[GrammarRule]:
    """Extract grammar rules from known secondary structure.

    Args:
        sequence: Amino acid sequence
        structure: Secondary structure string (H=helix, E=sheet, C=coil)

    Returns:
        List of derived grammar rules
    """
    rules = []

    # Find helix regions
    for match in re.finditer(r"H{4,}", structure):
        start, end = match.span()
        helix_seq = sequence[start:end]

        rules.append(
            GrammarRule(
                name=f"helix_{start}",
                lhs="HELIX",
                rhs=list(helix_seq),
                pattern=helix_seq,
                motif_type=MotifType.STRUCTURAL,
            )
        )

    # Find sheet regions
    for match in re.finditer(r"E{3,}", structure):
        start, end = match.span()
        sheet_seq = sequence[start:end]

        rules.append(
            GrammarRule(
                name=f"sheet_{start}",
                lhs="SHEET",
                rhs=list(sheet_seq),
                pattern=sheet_seq,
                motif_type=MotifType.STRUCTURAL,
            )
        )

    return rules


__all__ = [
    "MotifType",
    "ProteinMotif",
    "GrammarRule",
    "ParseTree",
    "PeptideGrammar",
    "MotifParser",
    "extract_secondary_structure_grammar",
]
