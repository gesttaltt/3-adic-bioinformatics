# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Unified Resistance Analyzer combining neural and set-theoretic approaches.

This module provides a comprehensive resistance analysis pipeline that integrates:
- Neural predictions (MTL, uncertainty quantification)
- Set-theoretic analysis (mutation sets, rough sets, lattices)
- Formal concept analysis (explainability, rule extraction)

The unified approach provides:
1. Accurate predictions via neural models
2. Uncertainty quantification via rough sets + neural uncertainty
3. Hierarchical classification via resistance lattice
4. Explainable results via FCA implication rules
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
import torch.nn as nn

from src.analysis.set_theory.formal_concepts import (
    FormalConcept,
    GenotypePhenotypeAnalyzer,
    ImplicationMiner,
    ImplicationRule,
)
from src.analysis.set_theory.lattice import (
    LatticeNode,
    ResistanceLattice,
    ResistanceLevel,
)
from src.analysis.set_theory.mutation_sets import (
    CrossResistanceAnalyzer,
    MutationSet,
    ResistanceProfile,
)
from src.analysis.set_theory.rough_sets import (
    RoughClassifier,
)


class DecisionType(Enum):
    """Three-way decision types."""

    ACCEPT = "accept"  # Definitely resistant
    REJECT = "reject"  # Definitely susceptible
    DEFER = "defer"  # Uncertain, needs more information


@dataclass
class AnalysisResult:
    """Complete resistance analysis result.

    Combines predictions, uncertainty, hierarchy, and explanations
    into a single comprehensive result.
    """

    # Primary prediction
    prediction: dict[str, float]  # Drug -> probability
    predicted_class: dict[str, str]  # Drug -> resistant/susceptible

    # Uncertainty quantification
    neural_uncertainty: dict[str, float]  # From MC Dropout/Ensemble
    rough_classification: dict[str, str]  # From rough sets
    combined_confidence: dict[str, float]  # Merged confidence
    decision_type: dict[str, DecisionType]  # Three-way decision

    # Hierarchical classification
    resistance_level: ResistanceLevel
    lattice_position: LatticeNode | None = None

    # Explainability
    triggered_rules: list[ImplicationRule] = field(default_factory=list)
    supporting_concepts: list[FormalConcept] = field(default_factory=list)
    evidence_mutations: dict[str, list[str]] = field(default_factory=dict)

    # Recommendations
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "prediction": self.prediction,
            "predicted_class": self.predicted_class,
            "neural_uncertainty": self.neural_uncertainty,
            "rough_classification": self.rough_classification,
            "combined_confidence": self.combined_confidence,
            "decision_type": {k: v.value for k, v in self.decision_type.items()},
            "resistance_level": self.resistance_level.name,
            "triggered_rules": [str(r) for r in self.triggered_rules],
            "evidence_mutations": self.evidence_mutations,
            "recommendations": self.recommendations,
        }


@dataclass
class AnalyzerConfig:
    """Configuration for UnifiedResistanceAnalyzer.

    Attributes:
        drug_names: List of drugs to analyze
        uncertainty_threshold: Threshold for uncertain classification
        rough_lower_threshold: VPRS lower threshold
        rough_upper_threshold: VPRS upper threshold
        min_confidence: Minimum confidence for accept/reject decision
        use_cross_attention: Enable cross-drug attention in predictions
    """

    drug_names: list[str] = field(
        default_factory=lambda: [
            "INH",
            "RIF",
            "EMB",
            "PZA",
            "STR",
            "FQ",
            "AMK",
            "CAP",
            "KAN",
            "ETH",
        ]
    )
    uncertainty_threshold: float = 0.3
    rough_lower_threshold: float = 0.9
    rough_upper_threshold: float = 0.1
    min_confidence: float = 0.7
    use_cross_attention: bool = True


class UnifiedResistanceAnalyzer:
    """Unified analyzer combining neural and set-theoretic approaches.

    This is the main entry point for comprehensive resistance analysis.
    It orchestrates multiple analysis methods and combines their results.

    Example:
        >>> # Initialize components
        >>> rough_classifiers = {drug: RoughClassifier.from_evidence(...) for drug in drugs}
        >>> lattice = ResistanceLattice()
        >>> fca = GenotypePhenotypeAnalyzer(samples, resistance)
        >>>
        >>> # Create analyzer
        >>> analyzer = UnifiedResistanceAnalyzer(
        ...     rough_classifiers=rough_classifiers,
        ...     lattice=lattice,
        ...     fca_analyzer=fca,
        ... )
        >>>
        >>> # Analyze a sample
        >>> result = analyzer.analyze({
        ...     "mutations": ["rpoB_S450L", "katG_S315T"],
        ...     "sample_id": "sample1",
        ... })
        >>> print(result.resistance_level)  # ResistanceLevel.MDR
    """

    def __init__(
        self,
        config: AnalyzerConfig | None = None,
        rough_classifiers: dict[str, RoughClassifier] | None = None,
        lattice: ResistanceLattice | None = None,
        fca_analyzer: GenotypePhenotypeAnalyzer | None = None,
        neural_model: nn.Module | None = None,
        encoder: nn.Module | None = None,
    ):
        """Initialize unified analyzer.

        Args:
            config: Analyzer configuration
            rough_classifiers: Drug-specific rough classifiers
            lattice: Resistance lattice structure
            fca_analyzer: Formal concept analyzer
            neural_model: Neural resistance predictor (optional)
            encoder: Sequence encoder for neural predictions (optional)
        """
        self.config = config or AnalyzerConfig()
        self.rough_classifiers = rough_classifiers or {}
        self.lattice = lattice or ResistanceLattice()
        self.fca_analyzer = fca_analyzer
        self.neural_model = neural_model
        self.encoder = encoder

        # Cross-resistance analyzer
        self.cross_analyzer = CrossResistanceAnalyzer()

        # Implication miner (if FCA available)
        self.implications: list[ImplicationRule] = []
        if self.fca_analyzer:
            self._mine_implications()

    def _mine_implications(self):
        """Mine implication rules from FCA context."""
        if self.fca_analyzer:
            miner = ImplicationMiner(self.fca_analyzer.context)
            self.implications = miner.compute_implications()

    def analyze(
        self,
        sample: dict[str, Any],
        include_neural: bool = True,
        include_explanation: bool = True,
    ) -> AnalysisResult:
        """Perform complete resistance analysis.

        Args:
            sample: Sample data with 'mutations' key (list of mutation strings)
            include_neural: Whether to include neural predictions
            include_explanation: Whether to include FCA explanations

        Returns:
            Complete analysis result
        """
        # 1. Parse mutations
        mutation_strings = sample.get("mutations", [])
        mutations = MutationSet.from_strings(mutation_strings, sample.get("sample_id", ""))

        # 2. Rough set classification (per drug)
        rough_results = self._rough_classify(mutations)

        # 3. Neural predictions (if available)
        neural_results = {}
        if include_neural and self.neural_model and self.encoder:
            neural_results = self._neural_predict(sample)

        # 4. Lattice-based hierarchy
        resistance_level = self.lattice.resistance_level(mutations)
        lattice_node = self._get_lattice_position(mutations)

        # 5. Combine predictions
        combined = self._combine_predictions(rough_results, neural_results)

        # 6. FCA explanation (if available)
        explanation = {}
        triggered_rules = []
        supporting_concepts = []
        if include_explanation and self.fca_analyzer:
            explanation = self._generate_explanation(mutations)
            triggered_rules = explanation.get("triggered_rules", [])
            supporting_concepts = explanation.get("supporting_concepts", [])

        # 7. Generate recommendations
        recommendations = self._generate_recommendations(rough_results, resistance_level, combined)

        return AnalysisResult(
            prediction=combined["predictions"],
            predicted_class=combined["classes"],
            neural_uncertainty=neural_results.get("uncertainty", {}),
            rough_classification={drug: result["classification"] for drug, result in rough_results.items()},
            combined_confidence=combined["confidence"],
            decision_type=combined["decisions"],
            resistance_level=resistance_level,
            lattice_position=lattice_node,
            triggered_rules=triggered_rules,
            supporting_concepts=supporting_concepts,
            evidence_mutations=self._extract_evidence(mutations, rough_results),
            recommendations=recommendations,
        )

    def _rough_classify(
        self,
        mutations: MutationSet,
    ) -> dict[str, dict[str, Any]]:
        """Classify using rough sets for each drug.

        Args:
            mutations: Mutation set to classify

        Returns:
            Per-drug rough classification results
        """
        results = {}

        for drug in self.config.drug_names:
            if drug in self.rough_classifiers:
                classifier = self.rough_classifiers[drug]
                result = classifier.classify_detailed(mutations)
                results[drug] = result
            else:
                # Default to uncertain if no classifier
                results[drug] = {
                    "classification": "uncertain",
                    "confidence": 0.5,
                    "evidence_strength": "none",
                    "definite_resistance_mutations": [],
                    "possible_resistance_mutations": [],
                }

        return results

    def _neural_predict(
        self,
        sample: dict[str, Any],
    ) -> dict[str, Any]:
        """Get neural model predictions with uncertainty.

        Args:
            sample: Sample data

        Returns:
            Neural predictions and uncertainty
        """
        if self.neural_model is None or self.encoder is None:
            return {"predictions": {}, "uncertainty": {}}

        # Get sequence embedding
        sequence = sample.get("sequence", "")
        if not sequence:
            return {"predictions": {}, "uncertainty": {}}

        with torch.no_grad():
            # Encode sequence
            if hasattr(self.encoder, "encode"):
                embedding = self.encoder.encode(sequence)
            else:
                # Assume encoder takes tokenized input
                embedding = self.encoder(sequence)

            # Predict with uncertainty (if wrapper available)
            if hasattr(self.neural_model, "predict_with_uncertainty"):
                result = self.neural_model.predict_with_uncertainty(embedding)
                predictions = result.get("prediction", result.get("mean", torch.tensor([])))
                uncertainty = result.get("uncertainty", result.get("std", torch.tensor([])))
            else:
                predictions = self.neural_model(embedding)
                uncertainty = torch.zeros_like(predictions)

        # Convert to per-drug dict
        pred_dict = {}
        unc_dict = {}
        for i, drug in enumerate(self.config.drug_names):
            if i < len(predictions):
                pred_dict[drug] = float(predictions[i])
                unc_dict[drug] = float(uncertainty[i]) if i < len(uncertainty) else 0.0

        return {"predictions": pred_dict, "uncertainty": unc_dict}

    def _get_lattice_position(
        self,
        mutations: MutationSet,
    ) -> LatticeNode | None:
        """Find position in resistance lattice.

        Args:
            mutations: Mutation set

        Returns:
            Lattice node or None
        """
        key = mutations.to_frozenset()
        return self.lattice.nodes.get(key)

    def _combine_predictions(
        self,
        rough_results: dict[str, dict[str, Any]],
        neural_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Combine rough and neural predictions.

        Args:
            rough_results: Rough set classifications
            neural_results: Neural predictions

        Returns:
            Combined predictions with confidence
        """
        predictions = {}
        classes = {}
        confidence = {}
        decisions = {}

        neural_preds = neural_results.get("predictions", {})
        neural_unc = neural_results.get("uncertainty", {})

        for drug in self.config.drug_names:
            rough = rough_results.get(drug, {})
            rough_class = rough.get("classification", "uncertain")
            rough_conf = rough.get("confidence", 0.5)

            neural_pred = neural_preds.get(drug, 0.5)
            neural_u = neural_unc.get(drug, 0.5)

            # Combine predictions
            if neural_pred > 0:
                # Weighted combination based on uncertainty
                neural_weight = 1 - neural_u
                rough_weight = rough_conf

                # Map rough classification to probability
                rough_prob = {
                    "resistant": 0.9,
                    "susceptible": 0.1,
                    "uncertain": 0.5,
                }.get(rough_class, 0.5)

                total_weight = neural_weight + rough_weight
                if total_weight > 0:
                    combined_pred = (neural_pred * neural_weight + rough_prob * rough_weight) / total_weight
                else:
                    combined_pred = 0.5
            else:
                combined_pred = {
                    "resistant": 0.9,
                    "susceptible": 0.1,
                    "uncertain": 0.5,
                }.get(rough_class, 0.5)

            predictions[drug] = combined_pred
            classes[drug] = "resistant" if combined_pred > 0.5 else "susceptible"

            # Compute combined confidence
            combined_conf = self._compute_combined_confidence(rough_conf, neural_u, rough_class)
            confidence[drug] = combined_conf

            # Three-way decision
            decisions[drug] = self._make_three_way_decision(combined_pred, combined_conf, rough_class)

        return {
            "predictions": predictions,
            "classes": classes,
            "confidence": confidence,
            "decisions": decisions,
        }

    def _compute_combined_confidence(
        self,
        rough_conf: float,
        neural_uncertainty: float,
        rough_class: str,
    ) -> float:
        """Compute combined confidence from rough and neural sources.

        Args:
            rough_conf: Rough set confidence
            neural_uncertainty: Neural uncertainty (higher = less confident)
            rough_class: Rough classification

        Returns:
            Combined confidence score
        """
        # Neural confidence is inverse of uncertainty
        neural_conf = 1 - neural_uncertainty

        # If rough is uncertain, weight neural more heavily
        if rough_class == "uncertain":
            return 0.3 * rough_conf + 0.7 * neural_conf
        else:
            return 0.6 * rough_conf + 0.4 * neural_conf

    def _make_three_way_decision(
        self,
        prediction: float,
        confidence: float,
        rough_class: str,
    ) -> DecisionType:
        """Make three-way decision based on evidence.

        Args:
            prediction: Combined prediction probability
            confidence: Combined confidence
            rough_class: Rough set classification

        Returns:
            Decision type (accept/reject/defer)
        """
        # If both rough and neural agree with high confidence
        if confidence >= self.config.min_confidence:
            if prediction > 0.5 + self.config.uncertainty_threshold:
                return DecisionType.ACCEPT
            elif prediction < 0.5 - self.config.uncertainty_threshold:
                return DecisionType.REJECT

        # If rough set says uncertain or confidence is low
        if rough_class == "uncertain" or confidence < self.config.min_confidence:
            return DecisionType.DEFER

        # Edge cases: moderate confidence
        if prediction > 0.5:
            return DecisionType.ACCEPT
        else:
            return DecisionType.REJECT

    def _generate_explanation(
        self,
        mutations: MutationSet,
    ) -> dict[str, Any]:
        """Generate FCA-based explanation.

        Args:
            mutations: Mutation set to explain

        Returns:
            Explanation with triggered rules and concepts
        """
        mutation_attrs = set(str(m) for m in mutations)

        # Find triggered implication rules
        triggered_rules = [rule for rule in self.implications if rule.applies_to(mutation_attrs)]

        # Find supporting concepts
        supporting_concepts = []
        if self.fca_analyzer:
            for concept in self.fca_analyzer.lattice.concepts:
                # Check if mutation set matches concept intent
                concept_mutations = {attr for attr in concept.intent if not attr.endswith("_R")}
                if concept_mutations.issubset(mutation_attrs):
                    supporting_concepts.append(concept)

        return {
            "triggered_rules": triggered_rules,
            "supporting_concepts": supporting_concepts,
            "n_rules_triggered": len(triggered_rules),
        }

    def _extract_evidence(
        self,
        mutations: MutationSet,
        rough_results: dict[str, dict[str, Any]],
    ) -> dict[str, list[str]]:
        """Extract evidence mutations for each drug.

        Args:
            mutations: Mutation set
            rough_results: Rough classification results

        Returns:
            Per-drug evidence mutations
        """
        evidence = {}

        for drug, result in rough_results.items():
            drug_evidence = []

            # Definite resistance mutations
            definite = result.get("definite_resistance_mutations", [])
            drug_evidence.extend([f"{m} (definite)" for m in definite])

            # Possible resistance mutations
            possible = result.get("possible_resistance_mutations", [])
            drug_evidence.extend([f"{m} (possible)" for m in possible])

            evidence[drug] = drug_evidence

        return evidence

    def _generate_recommendations(
        self,
        rough_results: dict[str, dict[str, Any]],
        resistance_level: ResistanceLevel,
        combined: dict[str, Any],
    ) -> list[str]:
        """Generate clinical recommendations.

        Args:
            rough_results: Rough classification results
            resistance_level: Overall resistance level
            combined: Combined predictions

        Returns:
            List of recommendations
        """
        recommendations = []

        # Deferred decisions need more testing
        deferred_drugs = [drug for drug, decision in combined["decisions"].items() if decision == DecisionType.DEFER]
        if deferred_drugs:
            recommendations.append(f"Additional testing recommended for: {', '.join(deferred_drugs)}")

        # Resistance level recommendations
        if resistance_level == ResistanceLevel.XDR:
            recommendations.append("XDR-TB detected. Consider specialized treatment regimen.")
        elif resistance_level == ResistanceLevel.PRE_XDR:
            recommendations.append("Pre-XDR-TB detected. Monitor for additional resistance development.")
        elif resistance_level == ResistanceLevel.MDR:
            recommendations.append("MDR-TB detected. Standard MDR regimen recommended.")

        # Low confidence warnings
        low_conf_drugs = [drug for drug, conf in combined["confidence"].items() if conf < 0.6]
        if low_conf_drugs:
            recommendations.append(
                f"Low confidence predictions for: {', '.join(low_conf_drugs)}. Consider phenotypic testing."
            )

        return recommendations

    def analyze_batch(
        self,
        samples: list[dict[str, Any]],
        include_neural: bool = True,
    ) -> list[AnalysisResult]:
        """Analyze multiple samples.

        Args:
            samples: List of sample data
            include_neural: Whether to include neural predictions

        Returns:
            List of analysis results
        """
        return [self.analyze(sample, include_neural=include_neural) for sample in samples]

    def build_from_training_data(
        self,
        samples: dict[str, list[str]],
        resistance: dict[str, list[str]],
        known_resistance_mutations: dict[str, dict[str, list[str]]] | None = None,
    ):
        """Build analyzer components from training data.

        Args:
            samples: Sample ID -> mutations mapping
            resistance: Sample ID -> resistant drugs mapping
            known_resistance_mutations: Drug -> {definite, possible} mutations
        """
        # Build FCA analyzer
        self.fca_analyzer = GenotypePhenotypeAnalyzer(samples, resistance)
        self._mine_implications()

        # Build rough classifiers
        if known_resistance_mutations:
            for drug, mutations in known_resistance_mutations.items():
                self.rough_classifiers[drug] = RoughClassifier.from_evidence(
                    definite_resistance=mutations.get("definite", []),
                    possible_resistance=mutations.get("possible", []),
                    drug_name=drug,
                )

        # Build resistance lattice from samples
        for sample_id, muts in samples.items():
            self.lattice.add_profile(sample_id, muts)

        # Add profiles to cross-resistance analyzer
        for sample_id, muts in samples.items():
            profile = ResistanceProfile(sample_id=sample_id)
            for drug in resistance.get(sample_id, []):
                profile.add_drug(drug, muts)
            self.cross_analyzer.add_profile(profile)

    def get_cross_resistance_matrix(self) -> dict[tuple[str, str], float]:
        """Get cross-resistance similarity matrix.

        Returns:
            Pairwise drug similarity scores
        """
        return self.cross_analyzer.similarity_matrix()

    def find_minimal_resistance_sets(
        self,
        drug: str,
    ) -> list[MutationSet]:
        """Find minimal mutation sets conferring resistance.

        Args:
            drug: Drug name

        Returns:
            List of minimal resistance-conferring mutation sets
        """
        if drug not in self.rough_classifiers:
            return []

        # Get all known resistance mutations for this drug
        classifier = self.rough_classifiers[drug]
        all_muts = MutationSet(classifier.positive_mutations.upper)

        return self.cross_analyzer.minimal_resistance_sets(drug, all_muts)
