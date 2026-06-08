# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Multi-disease framework for unified pathogen/disease modeling.

This module provides a disease-agnostic framework for training models
across multiple disease domains including:
- HIV (drug resistance, immune escape, glycan shielding)
- Rheumatoid Arthritis (citrullination, HLA interactions)
- Neurodegeneration (tau phosphorylation, amyloid aggregation)
- Cancer (neoantigen prediction, immune checkpoint)
- Infectious diseases (emerging pathogens, variant tracking)

Example:
    from src.diseases import DiseaseRegistry, get_disease_config

    # Get all registered diseases
    diseases = DiseaseRegistry.list_diseases()

    # Get specific disease config
    hiv_config = get_disease_config("hiv")

    # Train multi-disease model
    from src.diseases.training import MultiDiseaseTrainer
    trainer = MultiDiseaseTrainer(diseases=["hiv", "ra", "neuro"])
    trainer.train()
"""

from src.diseases.acinetobacter_analyzer import (
    ABClonalComplex,
    ABDrug,
    ABGene,
    AcinetobacterAnalyzer,
    AcinetobacterConfig,
)
from src.diseases.base import DiseaseConfig, DiseaseDataset
from src.diseases.cancer_analyzer import (
    CancerAnalyzer,
    CancerConfig,
    CancerGene,
    CancerType,
    TargetedTherapy,
    create_cancer_synthetic_dataset,
)
from src.diseases.candida_analyzer import (
    Antifungal,
    CandidaAnalyzer,
    CandidaClade,
    CandidaConfig,
    CandidaGene,
    create_candida_synthetic_dataset,
)
from src.diseases.cdiff_analyzer import (
    CDiffAnalyzer,
    CDiffConfig,
    CDiffDrug,
    CDiffGene,
    CDiffRibotype,
)
from src.diseases.dengue_analyzer import (
    DengueAnalyzer,
    DengueConfig,
    DengueDrug,
    DengueGene,
    DengueSerotype,
)
from src.diseases.ecoli_betalactam_analyzer import (
    TEM1_REFERENCE,
    TEM_MUTATIONS,
    BetaLactam,
    EcoliBetaLactamAnalyzer,
    EcoliBetaLactamConfig,
    EcoliGene,
    TEMVariant,
    create_ecoli_synthetic_dataset,
)
from src.diseases.gonorrhoeae_analyzer import (
    GCDrug,
    GCGene,
    GCSequenceType,
    GonorrhoeaeAnalyzer,
    GonorrhoeaeConfig,
)
from src.diseases.hbv_analyzer import (
    HBVAnalyzer,
    HBVConfig,
    HBVDrug,
    HBVGene,
    HBVGenotype,
    create_hbv_synthetic_dataset,
)
from src.diseases.hcv_analyzer import (
    HCVAnalyzer,
    HCVConfig,
    HCVDrug,
    HCVGene,
    HCVGenotype,
    create_hcv_synthetic_dataset,
)
from src.diseases.hiv_analyzer import (
    HIVAnalyzer,
    HIVConfig,
    HIVDrug,
    HIVDrugClass,
    HIVGene,
    create_hiv_synthetic_dataset,
)
from src.diseases.influenza_analyzer import (
    InfluenzaAnalyzer,
    InfluenzaConfig,
    InfluenzaDrug,
    InfluenzaGene,
    InfluenzaSubtype,
    create_influenza_synthetic_dataset,
)
from src.diseases.losses import MultiDiseaseLoss
from src.diseases.malaria_analyzer import (
    MalariaAnalyzer,
    MalariaConfig,
    MalariaDrug,
    MalariaGene,
    PlasmodiumSpecies,
    create_malaria_synthetic_dataset,
)
from src.diseases.mrsa_analyzer import (
    Antibiotic,
    MRSAAnalyzer,
    MRSAConfig,
    StaphGene,
    create_mrsa_synthetic_dataset,
)
from src.diseases.registry import DiseaseRegistry, get_disease_config
from src.diseases.rsv_analyzer import (
    RSVAnalyzer,
    RSVConfig,
    RSVDrug,
    RSVGene,
    RSVSubtype,
    create_rsv_synthetic_dataset,
)
from src.diseases.sars_cov2_analyzer import (
    SARSCoV2Analyzer,
    SARSCoV2Config,
    SARSCoV2Gene,
    SARSCoV2Variant,
    create_sars_cov2_dataset,
)
from src.diseases.tuberculosis_analyzer import (
    ResistanceLevel,
    TBDrug,
    TBGene,
    TuberculosisAnalyzer,
    TuberculosisConfig,
    create_tb_synthetic_dataset,
)
from src.diseases.uncertainty_aware_analyzer import (
    UncertaintyAwareAnalyzer,
    UncertaintyConfig,
    UncertaintyMethod,
    UncertaintyResult,
    create_uncertainty_analyzer,
)
from src.diseases.variant_escape import (
    DiseaseType,
    DrugResistancePredictor,
    EscapePrediction,
    FitnessPredictor,
    ImmuneEscapePredictor,
    MetaLearningEscapeHead,
    ReceptorBindingPredictor,
    VariantEscapeHead,
)
from src.diseases.vre_analyzer import (
    EnterococcusSpecies,
    VREAnalyzer,
    VREConfig,
    VREDrug,
    VREGene,
)
from src.diseases.zika_analyzer import (
    ZikaAnalyzer,
    ZikaConfig,
    ZikaDrug,
    ZikaGene,
    ZikaLineage,
)

__all__ = [
    "DiseaseRegistry",
    "get_disease_config",
    "DiseaseConfig",
    "DiseaseDataset",
    "MultiDiseaseLoss",
    # Variant escape prediction
    "VariantEscapeHead",
    "MetaLearningEscapeHead",
    "EscapePrediction",
    "DiseaseType",
    "FitnessPredictor",
    "ImmuneEscapePredictor",
    "DrugResistancePredictor",
    "ReceptorBindingPredictor",
    # SARS-CoV-2 specific
    "SARSCoV2Analyzer",
    "SARSCoV2Config",
    "SARSCoV2Gene",
    "SARSCoV2Variant",
    "create_sars_cov2_dataset",
    # Tuberculosis specific
    "TuberculosisAnalyzer",
    "TuberculosisConfig",
    "TBDrug",
    "TBGene",
    "ResistanceLevel",
    "create_tb_synthetic_dataset",
    # Influenza specific
    "InfluenzaAnalyzer",
    "InfluenzaConfig",
    "InfluenzaSubtype",
    "InfluenzaGene",
    "InfluenzaDrug",
    "create_influenza_synthetic_dataset",
    # HCV specific
    "HCVAnalyzer",
    "HCVConfig",
    "HCVGenotype",
    "HCVGene",
    "HCVDrug",
    "create_hcv_synthetic_dataset",
    # HBV specific
    "HBVAnalyzer",
    "HBVConfig",
    "HBVGenotype",
    "HBVGene",
    "HBVDrug",
    "create_hbv_synthetic_dataset",
    # Malaria specific
    "MalariaAnalyzer",
    "MalariaConfig",
    "PlasmodiumSpecies",
    "MalariaGene",
    "MalariaDrug",
    "create_malaria_synthetic_dataset",
    # MRSA specific
    "MRSAAnalyzer",
    "MRSAConfig",
    "StaphGene",
    "Antibiotic",
    "create_mrsa_synthetic_dataset",
    # Candida auris specific
    "CandidaAnalyzer",
    "CandidaConfig",
    "CandidaClade",
    "CandidaGene",
    "Antifungal",
    "create_candida_synthetic_dataset",
    # RSV specific
    "RSVAnalyzer",
    "RSVConfig",
    "RSVSubtype",
    "RSVGene",
    "RSVDrug",
    "create_rsv_synthetic_dataset",
    # Cancer targeted therapy
    "CancerAnalyzer",
    "CancerConfig",
    "CancerType",
    "CancerGene",
    "TargetedTherapy",
    "create_cancer_synthetic_dataset",
    # E. coli TEM beta-lactamase
    "EcoliBetaLactamAnalyzer",
    "EcoliBetaLactamConfig",
    "EcoliGene",
    "BetaLactam",
    "TEMVariant",
    "TEM_MUTATIONS",
    "TEM1_REFERENCE",
    "create_ecoli_synthetic_dataset",
    # HIV specific
    "HIVAnalyzer",
    "HIVConfig",
    "HIVGene",
    "HIVDrug",
    "HIVDrugClass",
    "create_hiv_synthetic_dataset",
    # Dengue Fever specific
    "DengueAnalyzer",
    "DengueConfig",
    "DengueSerotype",
    "DengueGene",
    "DengueDrug",
    # Zika Fever specific
    "ZikaAnalyzer",
    "ZikaConfig",
    "ZikaLineage",
    "ZikaGene",
    "ZikaDrug",
    # Clostridioides difficile specific
    "CDiffAnalyzer",
    "CDiffConfig",
    "CDiffRibotype",
    "CDiffGene",
    "CDiffDrug",
    # Neisseria gonorrhoeae specific
    "GonorrhoeaeAnalyzer",
    "GonorrhoeaeConfig",
    "GCSequenceType",
    "GCGene",
    "GCDrug",
    # Vancomycin-Resistant Enterococcus specific
    "VREAnalyzer",
    "VREConfig",
    "EnterococcusSpecies",
    "VREGene",
    "VREDrug",
    # Acinetobacter baumannii specific
    "AcinetobacterAnalyzer",
    "AcinetobacterConfig",
    "ABClonalComplex",
    "ABGene",
    "ABDrug",
    # Uncertainty-aware analysis
    "UncertaintyAwareAnalyzer",
    "UncertaintyConfig",
    "UncertaintyMethod",
    "UncertaintyResult",
    "create_uncertainty_analyzer",
]
