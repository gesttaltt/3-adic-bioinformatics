# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""FCA-based Explainability Module for Resistance Predictions.

Provides interpretable explanations for resistance predictions using
Formal Concept Analysis. Explanations include:
- Triggered implication rules (if mutation X → resistance Y)
- Supporting formal concepts (genotype clusters)
- Feature importance via attribute exploration
- Counterfactual explanations (what mutations would change prediction)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.analysis.set_theory.formal_concepts import (
    ConceptLattice,
    FormalConcept,
    GenotypePhenotypeAnalyzer,
    ImplicationMiner,
    ImplicationRule,
)
from src.analysis.set_theory.lattice import ResistanceLattice
from src.analysis.set_theory.mutation_sets import MutationSet
from src.analysis.set_theory.rough_sets import RoughClassifier


@dataclass
class Explanation:
    """Structured explanation for a resistance prediction.

    Attributes:
        prediction: The prediction being explained
        confidence: Prediction confidence
        triggered_rules: Implication rules that fired
        supporting_concepts: FCA concepts matching the sample
        key_mutations: Most important mutations for this prediction
        counterfactuals: Mutations that would change the prediction
        natural_language: Human-readable explanation
    """

    prediction: str
    confidence: float
    drug: str
    triggered_rules: list[ImplicationRule] = field(default_factory=list)
    supporting_concepts: list[FormalConcept] = field(default_factory=list)
    key_mutations: list[tuple[str, float]] = field(default_factory=list)
    counterfactuals: list[dict[str, Any]] = field(default_factory=list)
    natural_language: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "prediction": self.prediction,
            "confidence": self.confidence,
            "drug": self.drug,
            "triggered_rules": [str(r) for r in self.triggered_rules],
            "n_supporting_concepts": len(self.supporting_concepts),
            "key_mutations": self.key_mutations,
            "counterfactuals": self.counterfactuals,
            "explanation": self.natural_language,
        }


class FCAExplainer:
    """Generate explanations using Formal Concept Analysis.

    Uses FCA to provide interpretable explanations based on:
    1. Implication rules: logical relationships between mutations/phenotypes
    2. Formal concepts: clusters of samples with shared properties
    3. Attribute importance: which mutations contribute most

    Example:
        >>> explainer = FCAExplainer(fca_analyzer)
        >>> mutations = MutationSet.from_strings(["rpoB_S450L", "katG_S315T"])
        >>> explanation = explainer.explain("RIF", mutations, prediction="resistant")
        >>> print(explanation.natural_language)
    """

    def __init__(
        self,
        fca_analyzer: GenotypePhenotypeAnalyzer,
        rough_classifiers: dict[str, RoughClassifier] | None = None,
        lattice: ResistanceLattice | None = None,
    ):
        """Initialize FCA explainer.

        Args:
            fca_analyzer: Pre-built genotype-phenotype analyzer
            rough_classifiers: Optional rough classifiers for evidence strength
            lattice: Optional resistance lattice for hierarchy context
        """
        self.fca = fca_analyzer
        self.rough_classifiers = rough_classifiers or {}
        self.lattice = lattice

        # Mine implication rules
        self.miner = ImplicationMiner(self.fca.context)
        self.implications = self.miner.compute_implications()

        # Precompute approximate rules with different confidence thresholds
        self.approximate_rules = self.miner.find_approximate_rules(min_support=0.05, min_confidence=0.7)

    def explain(
        self,
        drug: str,
        mutations: MutationSet,
        prediction: str,
        confidence: float = 1.0,
    ) -> Explanation:
        """Generate explanation for a prediction.

        Args:
            drug: Drug name
            mutations: Observed mutations
            prediction: Model prediction ("resistant" or "susceptible")
            confidence: Prediction confidence

        Returns:
            Structured explanation
        """
        mutation_attrs = set(str(m) for m in mutations)
        resistance_attr = f"{drug}_R"

        # 1. Find triggered implication rules
        triggered_rules = self._find_triggered_rules(mutation_attrs, resistance_attr)

        # 2. Find supporting concepts
        supporting_concepts = self._find_supporting_concepts(mutation_attrs)

        # 3. Identify key mutations
        key_mutations = self._identify_key_mutations(drug, mutations)

        # 4. Generate counterfactuals
        counterfactuals = self._generate_counterfactuals(drug, mutations, prediction)

        # 5. Generate natural language explanation
        natural_language = self._generate_natural_language(drug, mutations, prediction, triggered_rules, key_mutations)

        return Explanation(
            prediction=prediction,
            confidence=confidence,
            drug=drug,
            triggered_rules=triggered_rules,
            supporting_concepts=supporting_concepts,
            key_mutations=key_mutations,
            counterfactuals=counterfactuals,
            natural_language=natural_language,
        )

    def _find_triggered_rules(
        self,
        mutation_attrs: set[str],
        resistance_attr: str,
    ) -> list[ImplicationRule]:
        """Find implication rules triggered by mutations.

        Args:
            mutation_attrs: Set of mutation attribute strings
            resistance_attr: Resistance phenotype attribute

        Returns:
            List of triggered rules
        """
        triggered = []

        for rule in self.implications + self.approximate_rules:
            # Check if antecedent is satisfied
            if rule.applies_to(mutation_attrs):
                # Check if consequent relates to resistance
                if resistance_attr in rule.consequent or not rule.consequent.isdisjoint(mutation_attrs):
                    triggered.append(rule)

        # Sort by confidence
        triggered.sort(key=lambda r: r.confidence, reverse=True)

        return triggered[:10]  # Top 10 rules

    def _find_supporting_concepts(
        self,
        mutation_attrs: set[str],
    ) -> list[FormalConcept]:
        """Find FCA concepts that match the mutation profile.

        Args:
            mutation_attrs: Mutation attributes

        Returns:
            Matching formal concepts
        """
        matching = []

        for concept in self.fca.lattice.concepts:
            # Get mutation-only attributes from concept intent
            concept_mutations = {attr for attr in concept.intent if not attr.endswith("_R")}

            # Check if concept mutations are subset of sample mutations
            if concept_mutations and concept_mutations.issubset(mutation_attrs):
                matching.append(concept)

        # Sort by specificity (more mutations = more specific)
        matching.sort(key=lambda c: len(c.intent), reverse=True)

        return matching[:5]  # Top 5 concepts

    def _identify_key_mutations(
        self,
        drug: str,
        mutations: MutationSet,
    ) -> list[tuple[str, float]]:
        """Identify mutations most important for this drug.

        Args:
            drug: Drug name
            mutations: Mutation set

        Returns:
            List of (mutation, importance) tuples
        """
        key_mutations = []

        # Use rough classifier if available
        if drug in self.rough_classifiers:
            classifier = self.rough_classifiers[drug]

            for mut in mutations:
                # Definite resistance mutation = high importance
                if classifier.positive_mutations.definitely_in(mut):
                    key_mutations.append((str(mut), 1.0))
                # Boundary (uncertain) = medium importance
                elif classifier.positive_mutations.uncertain(mut):
                    key_mutations.append((str(mut), 0.5))

        # Also use FCA association strength
        resistance_attr = f"{drug}_R"
        for mut in mutations:
            mut_str = str(mut)

            # Check how often this mutation co-occurs with resistance
            mut_extent = self.fca.context.attribute_extent({mut_str})
            res_extent = self.fca.context.attribute_extent({resistance_attr})

            if mut_extent and res_extent:
                # Jaccard-like association
                intersection = len(mut_extent & res_extent)
                union = len(mut_extent | res_extent)
                association = intersection / union if union > 0 else 0

                # Update or add
                existing = [m for m, _ in key_mutations if m == mut_str]
                if not existing:
                    key_mutations.append((mut_str, association))

        # Sort by importance
        key_mutations.sort(key=lambda x: x[1], reverse=True)

        return key_mutations

    def _generate_counterfactuals(
        self,
        drug: str,
        mutations: MutationSet,
        prediction: str,
    ) -> list[dict[str, Any]]:
        """Generate counterfactual explanations.

        What would need to change to flip the prediction?

        Args:
            drug: Drug name
            mutations: Current mutations
            prediction: Current prediction

        Returns:
            List of counterfactual scenarios
        """
        counterfactuals = []
        resistance_attr = f"{drug}_R"

        if prediction == "resistant":
            # What mutations to remove to become susceptible?
            key_muts = self._identify_key_mutations(drug, mutations)

            for mut_str, importance in key_muts[:3]:
                if importance > 0.5:
                    counterfactuals.append(
                        {
                            "action": "remove",
                            "mutation": mut_str,
                            "expected_change": "susceptible",
                            "confidence": importance,
                        }
                    )

        else:  # susceptible
            # What mutations would confer resistance?
            # Look at rules that have resistance in consequent
            for rule in self.implications:
                if resistance_attr in rule.consequent:
                    missing = rule.antecedent - set(str(m) for m in mutations)
                    if missing and len(missing) <= 2:
                        counterfactuals.append(
                            {
                                "action": "add",
                                "mutations": list(missing),
                                "expected_change": "resistant",
                                "confidence": rule.confidence,
                            }
                        )

        return counterfactuals[:5]

    def _generate_natural_language(
        self,
        drug: str,
        mutations: MutationSet,
        prediction: str,
        triggered_rules: list[ImplicationRule],
        key_mutations: list[tuple[str, float]],
    ) -> str:
        """Generate human-readable explanation.

        Args:
            drug: Drug name
            mutations: Mutation set
            prediction: Prediction
            triggered_rules: Triggered rules
            key_mutations: Key mutations

        Returns:
            Natural language explanation
        """
        lines = []

        # Header
        lines.append(f"Prediction: {prediction.upper()} to {drug}")
        lines.append("")

        # Key mutations
        if key_mutations:
            lines.append("Key mutations contributing to this prediction:")
            for mut, importance in key_mutations[:5]:
                strength = "strong" if importance > 0.7 else "moderate" if importance > 0.4 else "weak"
                lines.append(f"  - {mut} ({strength} evidence)")
            lines.append("")

        # Triggered rules
        if triggered_rules:
            lines.append("Implication rules supporting this prediction:")
            for rule in triggered_rules[:3]:
                ant = ", ".join(sorted(rule.antecedent)[:3])
                cons = ", ".join(sorted(rule.consequent)[:2])
                lines.append(f"  - If {{{ant}}} then {{{cons}}} (conf: {rule.confidence:.2f})")
            lines.append("")

        # Resistance level context
        if self.lattice:
            level = self.lattice.resistance_level(mutations)
            lines.append(f"Overall resistance level: {level.name}")

        return "\n".join(lines)

    def explain_batch(
        self,
        drug: str,
        mutation_sets: list[MutationSet],
        predictions: list[str],
        confidences: list[float] | None = None,
    ) -> list[Explanation]:
        """Explain a batch of predictions.

        Args:
            drug: Drug name
            mutation_sets: List of mutation sets
            predictions: List of predictions
            confidences: Optional confidence scores

        Returns:
            List of explanations
        """
        if confidences is None:
            confidences = [1.0] * len(predictions)

        return [
            self.explain(drug, ms, pred, conf)
            for ms, pred, conf in zip(mutation_sets, predictions, confidences, strict=False)
        ]


class RuleBasedValidator:
    """Validate predictions against known implication rules.

    Checks if neural network predictions are consistent with
    domain knowledge encoded in FCA implication rules.
    """

    def __init__(
        self,
        implications: list[ImplicationRule],
        strict: bool = False,
    ):
        """Initialize validator.

        Args:
            implications: List of implication rules
            strict: If True, flag any rule violation
        """
        self.implications = implications
        self.strict = strict

    def validate(
        self,
        mutations: MutationSet,
        predictions: dict[str, str],
    ) -> dict[str, Any]:
        """Validate predictions against rules.

        Args:
            mutations: Mutation set
            predictions: Drug -> prediction mapping

        Returns:
            Validation result with violations
        """
        mutation_attrs = set(str(m) for m in mutations)
        violations = []
        warnings = []

        for rule in self.implications:
            if rule.applies_to(mutation_attrs):
                # Rule applies - check if consequent is satisfied
                for attr in rule.consequent:
                    if attr.endswith("_R"):
                        drug = attr[:-2]  # Remove _R suffix

                        if drug in predictions:
                            # If rule says resistant but prediction is susceptible
                            if predictions[drug] == "susceptible":
                                if rule.confidence > 0.9:
                                    violations.append(
                                        {
                                            "rule": str(rule),
                                            "drug": drug,
                                            "expected": "resistant",
                                            "predicted": "susceptible",
                                            "confidence": rule.confidence,
                                        }
                                    )
                                else:
                                    warnings.append(
                                        {
                                            "rule": str(rule),
                                            "drug": drug,
                                            "message": "Rule suggests resistance but predicted susceptible",
                                        }
                                    )

        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "warnings": warnings,
            "n_rules_checked": len([r for r in self.implications if r.applies_to(mutation_attrs)]),
        }


class AttributeExplorer:
    """Explore attribute importance using FCA.

    Identifies which mutations are most predictive of resistance
    based on their position in the concept lattice.
    """

    def __init__(self, concept_lattice: ConceptLattice):
        """Initialize explorer.

        Args:
            concept_lattice: Concept lattice for exploration
        """
        self.lattice = concept_lattice

    def attribute_importance(
        self,
        target_attribute: str,
    ) -> dict[str, float]:
        """Compute importance of each attribute for predicting target.

        Args:
            target_attribute: Target attribute (e.g., "RIF_R")

        Returns:
            Attribute -> importance mapping
        """
        importance = {}

        # Find concepts containing target attribute
        target_concepts = self.lattice.attribute_concepts(target_attribute)

        if not target_concepts:
            return importance

        # For each other attribute, compute association
        for concept in self.lattice.concepts:
            for attr in concept.intent:
                if attr == target_attribute or attr.endswith("_R"):
                    continue

                # How often does this attribute appear with target?
                attr_concepts = self.lattice.attribute_concepts(attr)

                # Compute overlap
                overlap = len(set(target_concepts) & set(attr_concepts))
                total = len(attr_concepts) if attr_concepts else 1

                if attr not in importance:
                    importance[attr] = 0

                importance[attr] = max(importance[attr], overlap / total)

        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def find_minimal_predictors(
        self,
        target_attribute: str,
        max_size: int = 3,
    ) -> list[set[str]]:
        """Find minimal attribute sets that predict target.

        Args:
            target_attribute: Target to predict
            max_size: Maximum set size

        Returns:
            List of minimal predictor sets
        """
        predictors = []

        # Get introducing concept for target
        intro = self.lattice.introducing_concept(target_attribute)
        if not intro:
            return predictors

        # The intent of the introducing concept (minus target) predicts it
        predictor = {attr for attr in intro.intent if attr != target_attribute and not attr.endswith("_R")}

        if predictor and len(predictor) <= max_size:
            predictors.append(predictor)

        # Look for smaller predictors in subconcepts
        for concept in self.lattice.concepts:
            if target_attribute in concept.intent:
                pred = {attr for attr in concept.intent if attr != target_attribute and not attr.endswith("_R")}
                if pred and len(pred) <= max_size:
                    # Check if minimal
                    is_minimal = not any(existing < pred for existing in predictors)
                    if is_minimal:
                        predictors.append(pred)

        return predictors
