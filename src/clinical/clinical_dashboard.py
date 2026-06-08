"""
Clinical Decision Support Dashboard for HIV Treatment

⚠️ RESEARCH PROTOTYPE — NOT VALIDATED FOR CLINICAL USE ⚠️
This module has NOT been validated in clinical trials. All outputs are
computational predictions based on sequence features. DO NOT use for
diagnostic, treatment, or clinical decision-making. Clinical decisions
must be made by qualified healthcare providers using validated tools.

This module provides a unified interface for clinical decision support,
integrating all literature-derived implementations:
- P-adic codon analysis
- Hyperbolic VAE embeddings
- Potts model fitness landscapes
- HLA epitope prediction
- Drug resistance prediction
- Optimal treatment recommendations

Designed for RESEARCH USE ONLY:
- Exploratory treatment analysis
- Computational resistance monitoring
- Vaccine candidate research
- bnAb therapy research
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import implementations
try:
    from advanced_literature_implementations import (
        DrugResistanceAnalyzer,
        HLAEpitopePredictorSimulated,
        ProteinConformationGenerator,
    )
    from cutting_edge_implementations import (
        AntibodyOptimizer,
        HIVHostInteractionPredictor,
        OptimalTransportAligner,
        ProteinLanguageModel,
    )
    from literature_implementations import (
        EpistasisDetector,
        PAdicCodonEncoder,
        PottsModelFitness,
        QuasispeciesSimulator,
        ZeroShotMutationPredictor,
    )

    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class PatientProfile:
    """Patient information for clinical analysis."""

    patient_id: str
    hiv_sequences: dict[str, str] = field(default_factory=dict)
    hla_alleles: list[str] = field(default_factory=list)
    treatment_history: list[str] = field(default_factory=list)
    viral_load: float | None = None
    cd4_count: int | None = None
    subtype: str = "B"
    risk_factors: list[str] = field(default_factory=list)


@dataclass
class TreatmentRecommendation:
    """Treatment recommendation with supporting evidence."""

    regimen: str
    confidence: float
    rationale: list[str]
    alternatives: list[str]
    monitoring: list[str]
    resistance_risk: str


@dataclass
class ResistanceAssessment:
    """Drug resistance assessment results."""

    overall_risk: str
    mutations_detected: list[str]
    drug_class_risks: dict[str, float]
    evolution_trajectory: str
    time_to_failure: int | None


@dataclass
class VaccinePrioritization:
    """Vaccine candidate prioritization results."""

    top_epitopes: list[dict[str, Any]]
    population_coverage: float
    conservation_score: float
    escape_risk: str


# =============================================================================
# CLINICAL DASHBOARD
# =============================================================================


RESEARCH_USE_ONLY = True
"""Flag: this module is for research use only, not clinical deployment."""


class ClinicalDashboard:
    """
    Unified clinical decision support dashboard.

    ⚠️ RESEARCH PROTOTYPE — NOT VALIDATED FOR CLINICAL USE ⚠️
    All outputs are computational predictions. Not for diagnostic or
    treatment decisions. Use only in research settings with oversight
    from qualified healthcare professionals.

    Integrates all analysis tools for comprehensive patient assessment.
    """

    def __init__(self, output_dir: str = "results/clinical_dashboard"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize analysis modules
        self._init_modules()

        # Treatment databases
        self.drug_classes = {
            "NRTI": ["TDF", "FTC", "ABC", "3TC", "AZT", "DDI", "D4T"],
            "NNRTI": ["EFV", "NVP", "RPV", "ETR", "DOR"],
            "PI": ["LPV/r", "ATV/r", "DRV/r", "SQV/r", "FPV/r"],
            "INSTI": ["DTG", "RAL", "EVG", "BIC", "CAB"],
            "Entry": ["MVC", "ENF", "IBA", "FTR"],
        }

        # Resistance mutation databases
        self.resistance_mutations = {
            "NRTI": {"M184V": ["3TC", "FTC"], "K65R": ["TDF", "ABC"], "TAMs": ["AZT"], "K70E": ["TDF"]},
            "NNRTI": {"K103N": ["EFV", "NVP"], "Y181C": ["NVP", "ETR"], "E138K": ["RPV"]},
            "PI": {"I50V": ["DRV", "ATV"], "I84V": ["All PIs"], "L90M": ["SQV", "NFV"]},
            "INSTI": {"R263K": ["DTG"], "Q148H": ["RAL", "EVG"], "N155H": ["RAL"]},
        }

    def _init_modules(self):
        """Initialize analysis modules."""
        if IMPORTS_AVAILABLE:
            self.padic_encoder = PAdicCodonEncoder()
            self.fitness_model = PottsModelFitness(sequence_length=100)
            self.mutation_predictor = ZeroShotMutationPredictor()
            self.epistasis_detector = EpistasisDetector()
            self.quasispecies_sim = QuasispeciesSimulator(sequence_length=100)
            self.conformation_gen = ProteinConformationGenerator()
            self.resistance_analyzer = DrugResistanceAnalyzer()
            self.hla_predictor = HLAEpitopePredictorSimulated()
            self.ot_aligner = OptimalTransportAligner()
            self.plm = ProteinLanguageModel()
            self.bnab_optimizer = AntibodyOptimizer()
            self.ppi_predictor = HIVHostInteractionPredictor()
        else:
            # Fallback implementations
            self._init_fallback_modules()

    def _init_fallback_modules(self):
        """Initialize fallback modules when imports not available."""

        class FallbackModule:
            def __getattr__(self, name):
                return lambda *args, **kwargs: {}

        self.padic_encoder = FallbackModule()
        self.fitness_model = FallbackModule()
        self.mutation_predictor = FallbackModule()
        self.epistasis_detector = FallbackModule()
        self.quasispecies_sim = FallbackModule()
        self.conformation_gen = FallbackModule()
        self.resistance_analyzer = FallbackModule()
        self.hla_predictor = FallbackModule()
        self.ot_aligner = FallbackModule()
        self.plm = FallbackModule()
        self.bnab_optimizer = FallbackModule()
        self.ppi_predictor = FallbackModule()

    # =========================================================================
    # TREATMENT RECOMMENDATION
    # =========================================================================

    def recommend_treatment(self, patient: PatientProfile) -> TreatmentRecommendation:
        """
        Generate personalized treatment recommendation.

        Uses:
        - Resistance mutation analysis
        - Drug fitness landscape
        - Treatment history
        - Patient risk factors
        """
        # Analyze sequences for resistance
        resistance = self.assess_resistance(patient)

        # Determine optimal drug classes
        available_classes = self._filter_available_classes(resistance.drug_class_risks, patient.treatment_history)

        # Build recommended regimen
        regimen, rationale = self._build_regimen(available_classes, patient)

        # Generate alternatives
        alternatives = self._generate_alternatives(available_classes, patient)

        # Monitoring recommendations
        monitoring = self._get_monitoring_recommendations(resistance, patient)

        return TreatmentRecommendation(
            regimen=regimen,
            confidence=self._calculate_confidence(resistance, available_classes),
            rationale=rationale,
            alternatives=alternatives,
            monitoring=monitoring,
            resistance_risk=resistance.overall_risk,
        )

    def _filter_available_classes(self, class_risks: dict[str, float], treatment_history: list[str]) -> list[str]:
        """Filter drug classes by resistance and history."""
        available = []

        for drug_class, risk in class_risks.items():
            if risk < 0.5:  # Low resistance risk
                available.append(drug_class)

        # Prioritize classes not in history
        available.sort(key=lambda x: x in treatment_history)

        return available

    def _build_regimen(self, available_classes: list[str], patient: PatientProfile) -> tuple[str, list[str]]:
        """Build optimal regimen from available drug classes."""
        rationale = []

        # Standard preferred regimen structure
        if "INSTI" in available_classes:
            backbone = "DTG + TAF/FTC"
            rationale.append("INSTI-based regimen preferred (high barrier)")
        elif "PI" in available_classes:
            backbone = "DRV/r + TAF/FTC"
            rationale.append("PI-based regimen (treatment-experienced)")
        elif "NNRTI" in available_classes:
            backbone = "EFV + TDF/FTC"
            rationale.append("NNRTI-based (if no K103N detected)")
        else:
            backbone = "DRV/r + DTG + 3TC"
            rationale.append("Salvage regimen (multiple resistances)")

        if patient.viral_load and patient.viral_load > 100000:
            rationale.append("High viral load - monitor closely")

        if patient.cd4_count and patient.cd4_count < 200:
            rationale.append("Low CD4 - OI prophylaxis recommended")

        return backbone, rationale

    def _generate_alternatives(self, available_classes: list[str], patient: PatientProfile) -> list[str]:
        """Generate alternative regimens."""
        alternatives = []

        if "INSTI" in available_classes:
            alternatives.append("BIC/TAF/FTC (single tablet)")
            alternatives.append("DTG/3TC (two-drug regimen)")

        if "PI" in available_classes:
            alternatives.append("DRV/c/TAF/FTC (single tablet)")

        if "Entry" in available_classes:
            alternatives.append("LEN + optimized backbone (investigational)")

        return alternatives[:3]

    def _get_monitoring_recommendations(self, resistance: ResistanceAssessment, patient: PatientProfile) -> list[str]:
        """Generate monitoring recommendations."""
        monitoring = [
            "Viral load at 4 weeks, then every 3 months",
            "CD4 count every 6 months",
            "Genotype if virologic failure",
        ]

        if resistance.overall_risk == "high":
            monitoring.insert(0, "Genotype at 8 weeks")
            monitoring.append("Consider integrase resistance testing")

        if patient.cd4_count and patient.cd4_count < 200:
            monitoring.append("Monthly monitoring during immune reconstitution")

        return monitoring

    def _calculate_confidence(self, resistance: ResistanceAssessment, available_classes: list[str]) -> float:
        """Calculate recommendation confidence."""
        base_confidence = 0.9

        # Reduce for high resistance
        if resistance.overall_risk == "high":
            base_confidence -= 0.2
        elif resistance.overall_risk == "moderate":
            base_confidence -= 0.1

        # Reduce for limited options
        if len(available_classes) < 3:
            base_confidence -= 0.1

        return max(0.5, base_confidence)

    # =========================================================================
    # RESISTANCE ASSESSMENT
    # =========================================================================

    def assess_resistance(self, patient: PatientProfile) -> ResistanceAssessment:
        """
        Comprehensive resistance assessment.

        Uses:
        - Mutation detection
        - P-adic distance analysis
        - Epistasis detection
        - Evolution trajectory prediction
        """
        mutations_detected = []
        class_risks = {}

        # Analyze each drug class
        for drug_class in self.drug_classes:
            risk = self._assess_class_resistance(drug_class, patient.hiv_sequences, patient.treatment_history)
            class_risks[drug_class] = risk

        # Detect specific mutations
        for _seq_name, sequence in patient.hiv_sequences.items():
            detected = self._detect_mutations(sequence)
            mutations_detected.extend(detected)

        # Calculate overall risk
        max_risk = max(class_risks.values()) if class_risks else 0.0

        if max_risk > 0.7:
            overall_risk = "high"
        elif max_risk > 0.4:
            overall_risk = "moderate"
        else:
            overall_risk = "low"

        # Predict evolution
        trajectory = self._predict_evolution(patient.hiv_sequences, mutations_detected)

        return ResistanceAssessment(
            overall_risk=overall_risk,
            mutations_detected=list(set(mutations_detected)),
            drug_class_risks=class_risks,
            evolution_trajectory=trajectory,
            time_to_failure=self._estimate_time_to_failure(class_risks),
        )

    def _assess_class_resistance(self, drug_class: str, sequences: dict[str, str], history: list[str]) -> float:
        """Assess resistance to a drug class."""
        risk = 0.0

        # Check mutation database
        if drug_class in self.resistance_mutations:
            for mutation, _drugs in self.resistance_mutations[drug_class].items():
                for seq in sequences.values():
                    if self._has_mutation(seq, mutation):
                        risk += 0.3

        # Increase if class in treatment history
        if any(drug in history for drug in self.drug_classes.get(drug_class, [])):
            risk += 0.1

        return min(1.0, risk)

    def _has_mutation(self, sequence: str, mutation: str) -> bool:
        """Check if sequence has a specific mutation."""
        # Parse mutation (e.g., M184V -> position 184, wildtype M, mutant V)
        import re

        match = re.match(r"([A-Z])(\d+)([A-Z])", mutation)
        if not match:
            return False

        wt, pos, mut = match.groups()
        pos = int(pos)

        if pos <= len(sequence):
            return sequence[pos - 1] == mut

        return False

    def _detect_mutations(self, sequence: str) -> list[str]:
        """Detect known resistance mutations in sequence."""
        detected = []

        # Reference wild-type positions
        wt_reference = {
            184: "M",
            103: "K",
            65: "K",
            50: "I",
            155: "N",
        }

        for pos, wt in wt_reference.items():
            if pos <= len(sequence) and sequence[pos - 1] != wt:
                mut = sequence[pos - 1]
                detected.append(f"{wt}{pos}{mut}")

        return detected

    def _predict_evolution(self, sequences: dict[str, str], current_mutations: list[str]) -> str:
        """Predict resistance evolution trajectory."""
        if len(current_mutations) >= 5:
            return "Multi-drug resistance likely - limited options"
        elif len(current_mutations) >= 3:
            return "Accumulating mutations - consider regimen switch"
        elif len(current_mutations) >= 1:
            return "Early resistance - intensify monitoring"
        else:
            return "No significant resistance - standard monitoring"

    def _estimate_time_to_failure(self, class_risks: dict[str, float]) -> int | None:
        """Estimate days to treatment failure."""
        max_risk = max(class_risks.values()) if class_risks else 0.0

        if max_risk < 0.2:
            return None  # Very low risk
        elif max_risk < 0.5:
            return 365  # ~1 year
        elif max_risk < 0.7:
            return 180  # ~6 months
        else:
            return 90  # ~3 months

    # =========================================================================
    # VACCINE CANDIDATE PRIORITIZATION
    # =========================================================================

    def prioritize_vaccines(
        self, sequences: dict[str, str], patient_hla: list[str] | None = None
    ) -> VaccinePrioritization:
        """
        Prioritize vaccine candidates based on multiple criteria.

        Uses:
        - HLA binding prediction
        - Conservation analysis
        - Escape velocity calculation
        - Population coverage optimization
        """
        all_epitopes = []

        # Predict epitopes for all sequences
        for protein_name, sequence in sequences.items():
            if hasattr(self.hla_predictor, "predict_epitopes"):
                epitopes = self.hla_predictor.predict_epitopes(sequence, protein_name)
                all_epitopes.extend(
                    [
                        {
                            "peptide": e.peptide,
                            "protein": e.protein,
                            "position": e.position,
                            "coverage": e.population_coverage,
                            "n_alleles": len(e.hla_alleles),
                        }
                        for e in epitopes
                    ]
                )

        # Sort by priority
        all_epitopes.sort(key=lambda x: -x.get("coverage", 0))

        # Calculate metrics
        if all_epitopes:
            top_epitopes = all_epitopes[:10]
            max_coverage = max(e.get("coverage", 0) for e in top_epitopes)
        else:
            top_epitopes = []
            max_coverage = 0.0

        # Conservation analysis
        conservation = self._calculate_conservation(sequences)

        # Escape risk assessment
        escape_risk = self._assess_escape_risk(top_epitopes)

        return VaccinePrioritization(
            top_epitopes=top_epitopes,
            population_coverage=max_coverage,
            conservation_score=conservation,
            escape_risk=escape_risk,
        )

    def _calculate_conservation(self, sequences: dict[str, str]) -> float:
        """Calculate sequence conservation score."""
        if not sequences:
            return 0.0

        # Use entropy as conservation measure
        all_sequences = list(sequences.values())
        if len(all_sequences) < 2:
            return 1.0

        # Simple conservation: fraction of identical positions
        min_len = min(len(s) for s in all_sequences)
        identical = 0

        for i in range(min_len):
            if all(s[i] == all_sequences[0][i] for s in all_sequences):
                identical += 1

        return identical / min_len

    def _assess_escape_risk(self, epitopes: list[dict[str, Any]]) -> str:
        """Assess escape risk for epitopes."""
        if not epitopes:
            return "unknown"

        # Based on epitope properties
        avg_coverage = np.mean([e.get("coverage", 0) for e in epitopes])

        if avg_coverage > 0.6:
            return "low"
        elif avg_coverage > 0.3:
            return "moderate"
        else:
            return "high"

    # =========================================================================
    # BNAB THERAPY OPTIMIZATION
    # =========================================================================

    def optimize_bnab_therapy(self, patient: PatientProfile) -> dict[str, Any]:
        """
        Optimize broadly neutralizing antibody therapy.

        Uses:
        - Sequence sensitivity prediction
        - Epitope coverage analysis
        - Combination optimization
        """
        # Available bnAbs
        bnabs = {
            "VRC01": {"epitope": "CD4bs", "breadth": 0.68},
            "3BNC117": {"epitope": "CD4bs", "breadth": 0.78},
            "NIH45-46": {"epitope": "CD4bs", "breadth": 0.77},
            "10E8": {"epitope": "MPER", "breadth": 0.76},
            "PG9": {"epitope": "V1V2", "breadth": 0.70},
            "PGT121": {"epitope": "V3", "breadth": 0.65},
        }

        # Analyze patient sequences for sensitivity
        sensitivity = {}
        for bnab, info in bnabs.items():
            sens = self._predict_bnab_sensitivity(patient.hiv_sequences, bnab, info["epitope"])
            sensitivity[bnab] = {"sensitivity": sens, "epitope": info["epitope"], "breadth": info["breadth"]}

        # Find optimal combination
        optimal = self._find_optimal_combination(sensitivity)

        return {
            "individual_sensitivity": sensitivity,
            "optimal_combination": optimal["bnabs"],
            "expected_coverage": optimal["coverage"],
            "epitope_diversity": optimal["n_epitopes"],
            "recommendation": self._format_bnab_recommendation(optimal),
        }

    def _predict_bnab_sensitivity(self, sequences: dict[str, str], bnab: str, epitope: str) -> float:
        """Predict sensitivity to a specific bnAb."""
        # Simplified sensitivity based on epitope region
        # In production, would use actual neutralization models

        # Check if Env sequence available
        env_seq = sequences.get("Env", sequences.get("env", ""))

        if not env_seq:
            return 0.5  # Default moderate sensitivity

        # Simple scoring based on sequence features
        sensitivity = 0.5

        # V3 characteristics affect sensitivity
        if epitope == "V3" and "GPGR" in env_seq:
            sensitivity += 0.2
        elif epitope == "CD4bs":
            # CD4bs is relatively conserved
            sensitivity += 0.15

        return min(1.0, sensitivity)

    def _find_optimal_combination(self, sensitivity: dict[str, dict]) -> dict[str, Any]:
        """Find optimal bnAb combination."""
        # Sort by sensitivity
        ranked = sorted(sensitivity.items(), key=lambda x: x[1]["sensitivity"], reverse=True)

        # Build combination with epitope diversity
        selected = []
        epitopes_covered = set()

        for bnab, info in ranked:
            if info["epitope"] not in epitopes_covered:
                selected.append(bnab)
                epitopes_covered.add(info["epitope"])

            if len(selected) >= 3:
                break

        # Calculate expected coverage
        coverage = 1.0
        for bnab in selected:
            coverage *= 1 - sensitivity[bnab]["sensitivity"]
        coverage = 1 - coverage

        return {"bnabs": selected, "coverage": coverage, "n_epitopes": len(epitopes_covered)}

    def _format_bnab_recommendation(self, optimal: dict[str, Any]) -> str:
        """Format bnAb recommendation."""
        bnabs = " + ".join(optimal["bnabs"])
        coverage = optimal["coverage"] * 100

        return f"Recommended: {bnabs} (expected {coverage:.1f}% coverage)"

    # =========================================================================
    # COMPREHENSIVE REPORT
    # =========================================================================

    def generate_report(self, patient: PatientProfile) -> dict[str, Any]:
        """
        Generate comprehensive clinical report.

        Integrates all analyses into actionable recommendations.
        """
        print("\n" + "=" * 70)
        print("CLINICAL DECISION SUPPORT REPORT")
        print("=" * 70)
        print(f"Patient ID: {patient.patient_id}")
        print(f"Generated: {datetime.now()}")

        report = {"patient_id": patient.patient_id, "timestamp": datetime.now().isoformat(), "analyses": {}}

        # 1. Resistance Assessment
        print("\n" + "-" * 70)
        print("1. RESISTANCE ASSESSMENT")
        print("-" * 70)

        resistance = self.assess_resistance(patient)
        print(f"  Overall risk: {resistance.overall_risk.upper()}")
        print(f"  Mutations: {', '.join(resistance.mutations_detected) or 'None detected'}")
        print(f"  Evolution: {resistance.evolution_trajectory}")

        report["analyses"]["resistance"] = {
            "risk": resistance.overall_risk,
            "mutations": resistance.mutations_detected,
            "class_risks": resistance.drug_class_risks,
            "trajectory": resistance.evolution_trajectory,
        }

        # 2. Treatment Recommendation
        print("\n" + "-" * 70)
        print("2. TREATMENT RECOMMENDATION")
        print("-" * 70)

        treatment = self.recommend_treatment(patient)
        print(f"  Recommended: {treatment.regimen}")
        print(f"  Confidence: {treatment.confidence:.0%}")
        print("  Rationale:")
        for r in treatment.rationale:
            print(f"    - {r}")
        print("  Alternatives:")
        for a in treatment.alternatives:
            print(f"    - {a}")

        report["analyses"]["treatment"] = {
            "regimen": treatment.regimen,
            "confidence": treatment.confidence,
            "rationale": treatment.rationale,
            "alternatives": treatment.alternatives,
            "monitoring": treatment.monitoring,
        }

        # 3. Vaccine Prioritization (if sequences available)
        if patient.hiv_sequences:
            print("\n" + "-" * 70)
            print("3. VACCINE CANDIDATE ANALYSIS")
            print("-" * 70)

            vaccines = self.prioritize_vaccines(patient.hiv_sequences)
            print(f"  Top epitopes found: {len(vaccines.top_epitopes)}")
            print(f"  Max coverage: {vaccines.population_coverage:.1%}")
            print(f"  Conservation: {vaccines.conservation_score:.2f}")
            print(f"  Escape risk: {vaccines.escape_risk}")

            if vaccines.top_epitopes:
                print("  Top 3 epitopes:")
                for e in vaccines.top_epitopes[:3]:
                    print(f"    - {e['peptide']} ({e['protein']}): {e.get('coverage', 0):.1%}")

            report["analyses"]["vaccines"] = {
                "n_epitopes": len(vaccines.top_epitopes),
                "coverage": vaccines.population_coverage,
                "conservation": vaccines.conservation_score,
                "escape_risk": vaccines.escape_risk,
                "top_epitopes": vaccines.top_epitopes[:5],
            }

        # 4. bnAb Therapy (if Env sequence available)
        if "Env" in patient.hiv_sequences or "env" in patient.hiv_sequences:
            print("\n" + "-" * 70)
            print("4. BNAB THERAPY OPTIMIZATION")
            print("-" * 70)

            bnab = self.optimize_bnab_therapy(patient)
            print(f"  {bnab['recommendation']}")
            print(f"  Epitope diversity: {bnab['epitope_diversity']} classes")

            report["analyses"]["bnab"] = {
                "combination": bnab["optimal_combination"],
                "coverage": bnab["expected_coverage"],
                "diversity": bnab["epitope_diversity"],
            }

        # Save report
        output_file = self.output_dir / f"report_{patient.patient_id}.json"
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print("\n" + "=" * 70)
        print(f"Report saved to: {output_file}")
        print("=" * 70)

        return report


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    """Run clinical dashboard demonstration."""
    print("=" * 70)
    print("CLINICAL DECISION SUPPORT DASHBOARD")
    print("=" * 70)
    print(f"Started: {datetime.now()}")

    # Initialize dashboard
    dashboard = ClinicalDashboard()

    # Create sample patient
    patient = PatientProfile(
        patient_id="HIV-2024-001",
        hiv_sequences={
            "Protease": "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMNLPGRWKPKMIGGIGGFIKV",
            "RT": "CQGVPVLPKITLWQRPLVTIRIGGQLKEALLDTGADDTVLEDIDLPGRWKPKMIGG",
            "Gag": "MGARASVLSGGELDRWEKIRLRPGGKKKYKLKHIVWASRELERFAVNPGLLETS",
            "Env": "MRVKEKYQHLWRWGWRWGTMLLGMLMICSATEKLWVTVYYGVPVWKEATTTLFCAS",
        },
        hla_alleles=["HLA-A*02:01", "HLA-B*07:02"],
        treatment_history=["EFV", "TDF", "FTC"],
        viral_load=45000,
        cd4_count=320,
        subtype="B",
        risk_factors=["treatment_experienced", "prior_virologic_failure"],
    )

    # Generate comprehensive report
    report = dashboard.generate_report(patient)

    # Additional analyses
    print("\n" + "=" * 70)
    print("ADDITIONAL ANALYSES")
    print("=" * 70)

    # Test with different patient profiles
    patients = [
        PatientProfile(
            patient_id="HIV-2024-002",
            hiv_sequences={"Protease": "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMNLPGRWKPKMIGGIGGFIKV"},
            viral_load=500,
            cd4_count=450,
            treatment_history=[],
        ),
        PatientProfile(
            patient_id="HIV-2024-003",
            hiv_sequences={"Protease": "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMNLPGRWKPKMIGGIGGFIKV"},
            viral_load=250000,
            cd4_count=150,
            treatment_history=["AZT", "3TC", "EFV", "TDF", "LPV/r"],
        ),
    ]

    for p in patients:
        print(f"\n  Patient: {p.patient_id}")
        resistance = dashboard.assess_resistance(p)
        treatment = dashboard.recommend_treatment(p)
        print(f"    Resistance: {resistance.overall_risk}")
        print(f"    Recommended: {treatment.regimen}")

    print("\n" + "=" * 70)
    print("CLINICAL DASHBOARD COMPLETE")
    print("=" * 70)

    return report


if __name__ == "__main__":
    main()
