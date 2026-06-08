"""
Validate All Literature Implementations

Quick validation script to ensure all implementations are working correctly.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path


def main():
    """Run validation of all implementations."""
    print("=" * 70)
    print("VALIDATING ALL LITERATURE IMPLEMENTATIONS")
    print("=" * 70)
    print(f"Started: {datetime.now()}")

    results = {}
    Path(__file__).parent

    # Test 1: Literature Implementations
    print("\n[1/4] Testing literature_implementations.py...")
    try:
        from literature_implementations import (
            EpistasisDetector,
            HyperbolicVAE,
            PAdicCodonEncoder,
            PersistentHomologyAnalyzer,
            PottsModelFitness,
            QuasispeciesSimulator,
            ZeroShotMutationPredictor,
        )

        # Quick tests
        encoder = PAdicCodonEncoder()
        assert encoder.encode("ATG") == 50
        print("  - PAdicCodonEncoder: OK")

        import torch

        vae = HyperbolicVAE(100, 64, 16)
        x = torch.randn(4, 100)
        out = vae(x)
        assert len(out) == 4
        print("  - HyperbolicVAE: OK")

        PottsModelFitness(10)
        print("  - PottsModelFitness: OK")

        ph = PersistentHomologyAnalyzer()
        stats = ph.persistence_statistics("MGARASVLS")
        assert "dim0_n_bars" in stats
        print("  - PersistentHomologyAnalyzer: OK")

        zs = ZeroShotMutationPredictor()
        effect = zs.predict_mutation_effect("MGARASVLS", 0, "M", "A")
        # Check for any of the expected keys
        assert any(k in effect for k in ["effect_class", "score", "predicted_effect"])
        print("  - ZeroShotMutationPredictor: OK")

        EpistasisDetector()
        print("  - EpistasisDetector: OK")

        QuasispeciesSimulator(10)
        print("  - QuasispeciesSimulator: OK")

        results["literature_implementations"] = "PASS"
        print("  [PASS] All literature implementations validated")

    except Exception as e:
        results["literature_implementations"] = f"FAIL: {e}"
        print(f"  [FAIL] {e}")

    # Test 2: Advanced Implementations
    print("\n[2/4] Testing advanced_literature_implementations.py...")
    try:
        from advanced_literature_implementations import (
            ConditionalFlowMatcher,
            DrugResistanceAnalyzer,
            HLAEpitopePredictorSimulated,
            ProteinConformationGenerator,
        )

        ConditionalFlowMatcher()
        print("  - ConditionalFlowMatcher: OK")

        pcg = ProteinConformationGenerator()
        ensemble = pcg.generate_ensemble("MGARASVLS", n_conformations=2)
        assert "conformations" in ensemble
        print("  - ProteinConformationGenerator: OK")

        dra = DrugResistanceAnalyzer()
        impact = dra.analyze_mutation_structural_impact("protease", "D30N")
        assert "binding_impact" in impact
        print("  - DrugResistanceAnalyzer: OK")

        hla = HLAEpitopePredictorSimulated()
        hla.predict_epitopes("MGARASVLSGGELDR", "Test")
        print("  - HLAEpitopePredictorSimulated: OK")

        results["advanced_implementations"] = "PASS"
        print("  [PASS] All advanced implementations validated")

    except Exception as e:
        results["advanced_implementations"] = f"FAIL: {e}"
        print(f"  [FAIL] {e}")

    # Test 3: Cutting Edge Implementations
    print("\n[3/4] Testing cutting_edge_implementations.py...")
    try:
        from cutting_edge_implementations import (
            AntibodyDiffusionModel,
            AntibodyOptimizer,
            HIVHostInteractionPredictor,
            OptimalTransportAligner,
            PPIGraphNeuralNetwork,
            ProteinLanguageModel,
        )

        ot = OptimalTransportAligner()
        d = ot.wasserstein_distance("MGARASVLS", "MGARASVLT")
        assert d >= 0
        print("  - OptimalTransportAligner: OK")

        plm = ProteinLanguageModel()
        emb = plm.get_embeddings("MGARASVLS")
        assert emb.shape[0] == 9
        print("  - ProteinLanguageModel: OK")

        AntibodyDiffusionModel()
        print("  - AntibodyDiffusionModel: OK")

        AntibodyOptimizer()
        print("  - AntibodyOptimizer: OK")

        PPIGraphNeuralNetwork()
        print("  - PPIGraphNeuralNetwork: OK")

        HIVHostInteractionPredictor()
        print("  - HIVHostInteractionPredictor: OK")

        results["cutting_edge_implementations"] = "PASS"
        print("  [PASS] All cutting-edge implementations validated")

    except Exception as e:
        results["cutting_edge_implementations"] = f"FAIL: {e}"
        print(f"  [FAIL] {e}")

    # Test 4: Clinical Dashboard
    print("\n[4/4] Testing clinical_dashboard.py...")
    try:
        from clinical_dashboard import (
            ClinicalDashboard,
            PatientProfile,
        )

        dashboard = ClinicalDashboard()

        patient = PatientProfile(
            patient_id="TEST-001",
            hiv_sequences={"Protease": "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLE"},
            viral_load=10000,
            cd4_count=400,
        )

        resistance = dashboard.assess_resistance(patient)
        assert hasattr(resistance, "overall_risk")
        print("  - ResistanceAssessment: OK")

        treatment = dashboard.recommend_treatment(patient)
        assert hasattr(treatment, "regimen")
        print("  - TreatmentRecommendation: OK")

        results["clinical_dashboard"] = "PASS"
        print("  [PASS] Clinical dashboard validated")

    except Exception as e:
        results["clinical_dashboard"] = f"FAIL: {e}"
        print(f"  [FAIL] {e}")

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    all_pass = True
    for module, status in results.items():
        status_str = "[PASS]" if status == "PASS" else f"[FAIL] {status}"
        print(f"  {module}: {status_str}")
        if status != "PASS":
            all_pass = False

    print("\n" + "=" * 70)
    if all_pass:
        print("ALL VALIDATIONS PASSED")
    else:
        print("SOME VALIDATIONS FAILED - CHECK ABOVE")
    print("=" * 70)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
