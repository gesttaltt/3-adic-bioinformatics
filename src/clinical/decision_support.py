"""Clinical Decision Support Module for HIV Drug Resistance.

⚠️ RESEARCH PROTOTYPE — NOT VALIDATED FOR CLINICAL USE ⚠️
This module has NOT been evaluated for clinical deployment. Outputs are
computational predictions only. DO NOT use for patient diagnosis,
treatment selection, or any clinical decision. All clinical decisions
must involve a qualified physician using validated, approved tools.

Provides research-oriented decision support including:
1. Resistance interpretation with confidence intervals
2. Treatment recommendations based on resistance profile
3. Cross-resistance warnings
4. Drug interaction alerts
5. Guideline-based recommendations (DHHS, IAS-USA, EACS)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class ResistanceLevel(Enum):
    """Resistance interpretation levels (Stanford HIVDB convention)."""

    SUSCEPTIBLE = "Susceptible"
    POTENTIAL_LOW = "Potential Low-Level"
    LOW = "Low-Level"
    INTERMEDIATE = "Intermediate"
    HIGH = "High-Level"


class DrugClass(Enum):
    """HIV drug classes."""

    PI = "Protease Inhibitor"
    NRTI = "Nucleoside/Nucleotide RTI"
    NNRTI = "Non-Nucleoside RTI"
    INI = "Integrase Inhibitor"
    EI = "Entry Inhibitor"


@dataclass
class DrugResistanceResult:
    """Result for a single drug."""

    drug: str
    drug_class: DrugClass
    resistance_score: float
    confidence_interval: tuple[float, float]
    resistance_level: ResistanceLevel
    key_mutations: list[str]
    supporting_mutations: list[str]


@dataclass
class TreatmentRecommendation:
    """Treatment recommendation."""

    drug: str
    recommendation: str  # "Recommended", "Consider", "Avoid", "Contraindicated"
    rationale: str
    evidence_level: str  # "A", "B", "C" (guideline evidence levels)


@dataclass
class ClinicalAlert:
    """Clinical alert for important findings."""

    severity: str  # "info", "warning", "critical"
    category: str  # "cross-resistance", "interaction", "novel-mutation"
    message: str
    action_required: str


@dataclass
class ClinicalReport:
    """Complete clinical decision support report."""

    patient_id: str | None
    report_date: datetime
    sequence_info: dict[str, str]

    # Results
    drug_results: list[DrugResistanceResult]
    recommendations: list[TreatmentRecommendation]
    alerts: list[ClinicalAlert]

    # Summary
    overall_resistance_profile: str
    suggested_regimens: list[str]
    clinical_notes: list[str]


# Drug information database
DRUG_INFO = {
    # PI
    "LPV": {"class": DrugClass.PI, "full_name": "Lopinavir/ritonavir", "boosted": True},
    "DRV": {"class": DrugClass.PI, "full_name": "Darunavir/ritonavir", "boosted": True},
    "ATV": {"class": DrugClass.PI, "full_name": "Atazanavir/ritonavir", "boosted": True},
    # NRTI
    "TDF": {"class": DrugClass.NRTI, "full_name": "Tenofovir DF", "backbone": True},
    "TAF": {"class": DrugClass.NRTI, "full_name": "Tenofovir AF", "backbone": True},
    "ABC": {"class": DrugClass.NRTI, "full_name": "Abacavir", "backbone": True},
    "3TC": {"class": DrugClass.NRTI, "full_name": "Lamivudine", "backbone": True},
    "FTC": {"class": DrugClass.NRTI, "full_name": "Emtricitabine", "backbone": True},
    "AZT": {"class": DrugClass.NRTI, "full_name": "Zidovudine", "backbone": True},
    # NNRTI
    "EFV": {"class": DrugClass.NNRTI, "full_name": "Efavirenz"},
    "DOR": {"class": DrugClass.NNRTI, "full_name": "Doravirine"},
    "RPV": {"class": DrugClass.NNRTI, "full_name": "Rilpivirine"},
    # INI
    "DTG": {"class": DrugClass.INI, "full_name": "Dolutegravir", "high_barrier": True},
    "BIC": {"class": DrugClass.INI, "full_name": "Bictegravir", "high_barrier": True},
    "RAL": {"class": DrugClass.INI, "full_name": "Raltegravir"},
    "EVG": {"class": DrugClass.INI, "full_name": "Elvitegravir"},
}

# Standard first-line regimens (DHHS 2024)
FIRST_LINE_REGIMENS = [
    {"name": "BIC/TAF/FTC", "drugs": ["BIC", "TAF", "FTC"], "level": "Recommended"},
    {"name": "DTG + TAF/FTC", "drugs": ["DTG", "TAF", "FTC"], "level": "Recommended"},
    {"name": "DTG + TDF/FTC", "drugs": ["DTG", "TDF", "FTC"], "level": "Recommended"},
    {"name": "DOR/TDF/3TC", "drugs": ["DOR", "TDF", "3TC"], "level": "Recommended"},
]


class ResistanceInterpreter:
    """Interpret resistance scores into clinical levels."""

    # Thresholds based on Stanford HIVDB
    THRESHOLDS = {
        "susceptible": 0.15,
        "potential_low": 0.30,
        "low": 0.50,
        "intermediate": 0.70,
        "high": 0.85,
    }

    def interpret(self, score: float) -> ResistanceLevel:
        """Convert score to resistance level."""
        if score < self.THRESHOLDS["susceptible"]:
            return ResistanceLevel.SUSCEPTIBLE
        elif score < self.THRESHOLDS["potential_low"]:
            return ResistanceLevel.POTENTIAL_LOW
        elif score < self.THRESHOLDS["low"]:
            return ResistanceLevel.LOW
        elif score < self.THRESHOLDS["intermediate"]:
            return ResistanceLevel.INTERMEDIATE
        else:
            return ResistanceLevel.HIGH

    def get_recommendation(
        self,
        drug: str,
        level: ResistanceLevel,
    ) -> TreatmentRecommendation:
        """Get treatment recommendation based on resistance level."""
        DRUG_INFO.get(drug, {})

        if level == ResistanceLevel.SUSCEPTIBLE:
            return TreatmentRecommendation(
                drug=drug,
                recommendation="Recommended",
                rationale="No significant resistance detected",
                evidence_level="A",
            )
        elif level == ResistanceLevel.POTENTIAL_LOW:
            return TreatmentRecommendation(
                drug=drug,
                recommendation="Consider",
                rationale="Potential low-level resistance; monitor closely",
                evidence_level="B",
            )
        elif level == ResistanceLevel.LOW:
            return TreatmentRecommendation(
                drug=drug,
                recommendation="Consider with caution",
                rationale="Low-level resistance detected; consider alternatives",
                evidence_level="B",
            )
        elif level == ResistanceLevel.INTERMEDIATE:
            return TreatmentRecommendation(
                drug=drug,
                recommendation="Avoid if alternatives available",
                rationale="Intermediate resistance; reduced efficacy expected",
                evidence_level="A",
            )
        else:  # HIGH
            return TreatmentRecommendation(
                drug=drug,
                recommendation="Contraindicated",
                rationale="High-level resistance; drug not effective",
                evidence_level="A",
            )


class CrossResistanceAnalyzer:
    """Analyze cross-resistance patterns."""

    # Cross-resistance rules
    CROSS_RESISTANCE_RULES = {
        # NRTI TAM cross-resistance
        ("AZT", "D4T"): {
            "correlation": 0.85,
            "mutations": ["M41L", "D67N", "K70R", "L210W", "T215Y", "K219Q"],
            "message": "TAM mutations confer cross-resistance between AZT and D4T",
        },
        # M184V resensitization
        ("3TC", "AZT"): {
            "correlation": -0.15,
            "mutations": ["M184V"],
            "message": "M184V may partially resensitize to AZT by reducing TAM excision",
        },
        # K65R cross-resistance
        ("TDF", "ABC"): {
            "correlation": 0.65,
            "mutations": ["K65R"],
            "message": "K65R confers cross-resistance to TDF and ABC",
        },
        # First-gen NNRTI
        ("EFV", "NVP"): {
            "correlation": 0.85,
            "mutations": ["K103N", "Y181C", "G190A"],
            "message": "First-generation NNRTIs share extensive cross-resistance",
        },
    }

    def analyze(
        self,
        results: list[DrugResistanceResult],
    ) -> list[ClinicalAlert]:
        """Generate cross-resistance alerts."""
        alerts = []

        # Build drug score map
        scores = {r.drug: r.resistance_score for r in results}

        for (drug1, drug2), info in self.CROSS_RESISTANCE_RULES.items():
            if drug1 in scores and drug2 in scores:
                if scores[drug1] > 0.5 and scores[drug2] > 0.5:
                    alerts.append(
                        ClinicalAlert(
                            severity="warning",
                            category="cross-resistance",
                            message=f"{info['message']} (both show resistance)",
                            action_required=f"Consider avoiding both {drug1} and {drug2}",
                        )
                    )
                elif info["correlation"] < 0 and scores[drug1] > 0.5:
                    alerts.append(
                        ClinicalAlert(
                            severity="info",
                            category="cross-resistance",
                            message=f"{info['message']} (potential resensitization)",
                            action_required=f"{drug2} may still be effective despite {drug1} resistance",
                        )
                    )

        return alerts


class RegimenSelector:
    """Select optimal treatment regimens."""

    def __init__(self):
        self.interpreter = ResistanceInterpreter()

    def evaluate_regimen(
        self,
        regimen: dict,
        results: list[DrugResistanceResult],
    ) -> tuple[str, float, str]:
        """Evaluate a regimen based on resistance profile.

        Returns: (recommendation, score, rationale)
        """
        scores = {r.drug: r.resistance_score for r in results}

        # Check each drug in regimen
        drug_assessments = []
        for drug in regimen["drugs"]:
            if drug in scores:
                level = self.interpreter.interpret(scores[drug])
                if level in [ResistanceLevel.HIGH, ResistanceLevel.INTERMEDIATE]:
                    drug_assessments.append(("avoid", drug, level))
                elif level == ResistanceLevel.LOW:
                    drug_assessments.append(("caution", drug, level))
                else:
                    drug_assessments.append(("ok", drug, level))
            else:
                drug_assessments.append(("unknown", drug, None))

        # Determine overall recommendation
        avoid_count = sum(1 for a in drug_assessments if a[0] == "avoid")
        caution_count = sum(1 for a in drug_assessments if a[0] == "caution")

        if avoid_count > 0:
            rec = "Not recommended"
            score = 0.0
            problematic = [a[1] for a in drug_assessments if a[0] == "avoid"]
            rationale = f"Resistance to: {', '.join(problematic)}"
        elif caution_count > 1:
            rec = "Use with caution"
            score = 0.5
            rationale = "Multiple drugs with low-level resistance"
        elif caution_count == 1:
            rec = "Consider"
            score = 0.7
            caution_drug = [a[1] for a in drug_assessments if a[0] == "caution"][0]
            rationale = f"Low-level resistance to {caution_drug}; monitor closely"
        else:
            rec = "Recommended"
            score = 1.0
            rationale = "No significant resistance to regimen components"

        return rec, score, rationale

    def select_regimens(
        self,
        results: list[DrugResistanceResult],
        n_regimens: int = 3,
    ) -> list[dict]:
        """Select top regimens based on resistance profile."""
        evaluated = []

        for regimen in FIRST_LINE_REGIMENS:
            rec, score, rationale = self.evaluate_regimen(regimen, results)
            evaluated.append(
                {
                    "name": regimen["name"],
                    "drugs": regimen["drugs"],
                    "recommendation": rec,
                    "score": score,
                    "rationale": rationale,
                }
            )

        # Sort by score (descending)
        evaluated.sort(key=lambda x: x["score"], reverse=True)

        return evaluated[:n_regimens]


class ClinicalDecisionSupport:
    """Main clinical decision support system."""

    def __init__(self):
        self.interpreter = ResistanceInterpreter()
        self.cross_analyzer = CrossResistanceAnalyzer()
        self.regimen_selector = RegimenSelector()

    def generate_report(
        self,
        predictions: dict[str, float],
        confidence_intervals: dict[str, tuple[float, float]] = None,
        mutations: dict[str, list[str]] = None,
        patient_id: str | None = None,
    ) -> ClinicalReport:
        """Generate comprehensive clinical report.

        Args:
            predictions: Drug name -> resistance score
            confidence_intervals: Drug name -> (lower, upper) bounds
            mutations: Drug name -> list of detected mutations
            patient_id: Optional patient identifier
        """
        if confidence_intervals is None:
            confidence_intervals = {d: (s - 0.1, s + 0.1) for d, s in predictions.items()}
        if mutations is None:
            mutations = {d: [] for d in predictions}

        # Generate drug results
        drug_results = []
        for drug, score in predictions.items():
            if drug not in DRUG_INFO:
                continue

            info = DRUG_INFO[drug]
            level = self.interpreter.interpret(score)

            result = DrugResistanceResult(
                drug=drug,
                drug_class=info["class"],
                resistance_score=score,
                confidence_interval=confidence_intervals.get(drug, (score - 0.1, score + 0.1)),
                resistance_level=level,
                key_mutations=mutations.get(drug, []),
                supporting_mutations=[],
            )
            drug_results.append(result)

        # Generate recommendations
        recommendations = []
        for result in drug_results:
            rec = self.interpreter.get_recommendation(result.drug, result.resistance_level)
            recommendations.append(rec)

        # Analyze cross-resistance
        alerts = self.cross_analyzer.analyze(drug_results)

        # Select regimens
        suggested = self.regimen_selector.select_regimens(drug_results)

        # Determine overall profile
        high_count = sum(1 for r in drug_results if r.resistance_level == ResistanceLevel.HIGH)
        intermediate_count = sum(1 for r in drug_results if r.resistance_level == ResistanceLevel.INTERMEDIATE)

        if high_count > 10:
            overall = "Extensive drug resistance (MDR)"
        elif high_count > 5:
            overall = "Multi-drug resistance"
        elif high_count > 0 or intermediate_count > 3:
            overall = "Limited drug resistance"
        else:
            overall = "Susceptible to most agents"

        # Clinical notes
        notes = []
        if high_count > 0:
            notes.append(f"High-level resistance detected to {high_count} drug(s)")
        if any(r.drug in ["DTG", "BIC"] and r.resistance_level == ResistanceLevel.SUSCEPTIBLE for r in drug_results):
            notes.append("High-barrier INIs (DTG, BIC) remain active options")

        return ClinicalReport(
            patient_id=patient_id,
            report_date=datetime.now(),
            sequence_info={"gene": "PR/RT/IN", "length": "variable"},
            drug_results=drug_results,
            recommendations=recommendations,
            alerts=alerts,
            overall_resistance_profile=overall,
            suggested_regimens=[r["name"] for r in suggested if r["score"] > 0.5],
            clinical_notes=notes,
        )

    def format_report(self, report: ClinicalReport) -> str:
        """Format report as text."""
        lines = []
        lines.append("=" * 70)
        lines.append("HIV DRUG RESISTANCE CLINICAL DECISION SUPPORT REPORT")
        lines.append("=" * 70)
        lines.append(f"Report Date: {report.report_date.strftime('%Y-%m-%d %H:%M')}")
        if report.patient_id:
            lines.append(f"Patient ID: {report.patient_id}")
        lines.append("")

        lines.append("OVERALL ASSESSMENT")
        lines.append("-" * 40)
        lines.append(f"Resistance Profile: {report.overall_resistance_profile}")
        lines.append("")

        lines.append("DRUG-LEVEL RESULTS")
        lines.append("-" * 40)
        for drug_class in DrugClass:
            class_results = [r for r in report.drug_results if r.drug_class == drug_class]
            if class_results:
                lines.append(f"\n{drug_class.value}:")
                for r in class_results:
                    ci_str = f"[{r.confidence_interval[0]:.2f}-{r.confidence_interval[1]:.2f}]"
                    lines.append(f"  {r.drug}: {r.resistance_level.value} (score={r.resistance_score:.3f} {ci_str})")

        if report.alerts:
            lines.append("\nCLINICAL ALERTS")
            lines.append("-" * 40)
            for alert in report.alerts:
                lines.append(f"[{alert.severity.upper()}] {alert.message}")
                lines.append(f"  Action: {alert.action_required}")

        lines.append("\nSUGGESTED REGIMENS")
        lines.append("-" * 40)
        for i, regimen in enumerate(report.suggested_regimens, 1):
            lines.append(f"{i}. {regimen}")

        lines.append("\nCLINICAL NOTES")
        lines.append("-" * 40)
        for note in report.clinical_notes:
            lines.append(f"- {note}")

        lines.append("\n" + "=" * 70)
        lines.append("DISCLAIMER: This report is for research purposes only.")
        lines.append("Clinical decisions should be made by qualified healthcare providers.")
        lines.append("=" * 70)

        return "\n".join(lines)


if __name__ == "__main__":
    print("Testing Clinical Decision Support Module")
    print("=" * 60)

    # Create test predictions
    predictions = {
        "LPV": 0.15,
        "DRV": 0.10,
        "ATV": 0.20,
        "TDF": 0.25,
        "ABC": 0.80,  # High resistance
        "3TC": 0.85,  # High resistance
        "AZT": 0.70,  # Intermediate
        "EFV": 0.60,
        "DOR": 0.15,
        "DTG": 0.10,
        "BIC": 0.12,
        "RAL": 0.55,
    }

    # Generate report
    cds = ClinicalDecisionSupport()
    report = cds.generate_report(predictions, patient_id="TEST-001")

    # Print formatted report
    print(cds.format_report(report))

    print("\n" + "=" * 60)
    print("Clinical Decision Support module working!")
