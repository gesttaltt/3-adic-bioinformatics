# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Drug-Drug Interaction Checker for Antimicrobial Therapy.

Provides comprehensive drug-drug interaction (DDI) checking for:
- HIV antiretrovirals
- Antibiotics (beta-lactams, aminoglycosides, fluoroquinolones, etc.)
- Antifungals (azoles, echinocandins, polyenes)
- Antivirals (HCV DAAs, HBV NAs, influenza NAIs)
- Antimalarials
- Cancer targeted therapies

Features:
- Severity classification (contraindicated, major, moderate, minor)
- Mechanism-based interaction explanations
- Clinical recommendations
- CYP450 interaction modeling
- QT prolongation risk assessment
- Nephrotoxicity/hepatotoxicity synergy detection

Example:
    from src.clinical.drug_interactions import DrugInteractionChecker

    checker = DrugInteractionChecker()

    # Check single pair
    interaction = checker.check_interaction("ritonavir", "simvastatin")
    print(f"Severity: {interaction.severity}")  # "contraindicated"

    # Check full regimen
    regimen = ["tenofovir", "lamivudine", "dolutegravir", "fluconazole"]
    report = checker.check_regimen(regimen)
    print(report.summary)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class InteractionSeverity(Enum):
    """Drug interaction severity levels."""

    CONTRAINDICATED = "contraindicated"  # Must not be used together
    MAJOR = "major"  # Serious clinical consequences, avoid
    MODERATE = "moderate"  # May require dose adjustment/monitoring
    MINOR = "minor"  # Minimal clinical significance
    NONE = "none"  # No known interaction


class InteractionMechanism(Enum):
    """Mechanism of drug-drug interaction."""

    CYP3A4_INHIBITION = "cyp3a4_inhibition"
    CYP3A4_INDUCTION = "cyp3a4_induction"
    CYP2D6_INHIBITION = "cyp2d6_inhibition"
    CYP2C19_INHIBITION = "cyp2c19_inhibition"
    CYP2C9_INHIBITION = "cyp2c9_inhibition"
    PGLYCOPROTEIN_INHIBITION = "p_glycoprotein_inhibition"
    PGLYCOPROTEIN_INDUCTION = "p_glycoprotein_induction"
    UGT_INHIBITION = "ugt_inhibition"
    QT_PROLONGATION = "qt_prolongation"
    NEPHROTOXICITY_SYNERGY = "nephrotoxicity_synergy"
    HEPATOTOXICITY_SYNERGY = "hepatotoxicity_synergy"
    BONE_MARROW_SUPPRESSION = "bone_marrow_suppression"
    PHARMACODYNAMIC = "pharmacodynamic"
    ABSORPTION_INTERACTION = "absorption_interaction"
    PROTEIN_BINDING = "protein_binding"
    RENAL_EXCRETION = "renal_excretion"


class DrugCategory(Enum):
    """Drug category for grouping."""

    # HIV
    HIV_PI = "hiv_protease_inhibitor"
    HIV_NRTI = "hiv_nrti"
    HIV_NNRTI = "hiv_nnrti"
    HIV_INI = "hiv_integrase_inhibitor"
    HIV_BOOSTER = "hiv_pharmacokinetic_booster"

    # Antibacterials
    BETA_LACTAM = "beta_lactam"
    AMINOGLYCOSIDE = "aminoglycoside"
    FLUOROQUINOLONE = "fluoroquinolone"
    GLYCOPEPTIDE = "glycopeptide"
    MACROLIDE = "macrolide"
    TETRACYCLINE = "tetracycline"
    OXAZOLIDINONE = "oxazolidinone"
    POLYMYXIN = "polymyxin"
    SULFONAMIDE = "sulfonamide"

    # Antifungals
    AZOLE_ANTIFUNGAL = "azole_antifungal"
    ECHINOCANDIN = "echinocandin"
    POLYENE = "polyene"

    # Antivirals
    HCV_DAA = "hcv_daa"
    HBV_NA = "hbv_na"
    INFLUENZA_NAI = "influenza_nai"

    # Other
    ANTIMALARIAL = "antimalarial"
    CANCER_TKI = "cancer_tki"
    IMMUNOSUPPRESSANT = "immunosuppressant"
    CARDIOVASCULAR = "cardiovascular"
    STATIN = "statin"


@dataclass
class DrugInfo:
    """Drug information for interaction checking."""

    name: str
    generic_name: str
    category: DrugCategory
    cyp3a4_substrate: bool = False
    cyp3a4_inhibitor: bool = False
    cyp3a4_inducer: bool = False
    cyp2d6_substrate: bool = False
    cyp2d6_inhibitor: bool = False
    cyp2c19_inhibitor: bool = False
    cyp2c9_inhibitor: bool = False
    pgp_substrate: bool = False
    pgp_inhibitor: bool = False
    pgp_inducer: bool = False
    qt_prolongation: bool = False
    nephrotoxic: bool = False
    hepatotoxic: bool = False
    myelosuppressive: bool = False
    aliases: list[str] = field(default_factory=list)


@dataclass
class Interaction:
    """Drug-drug interaction result."""

    drug1: str
    drug2: str
    severity: InteractionSeverity
    mechanism: InteractionMechanism
    description: str
    clinical_effect: str
    recommendation: str
    monitoring: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)


@dataclass
class RegimenReport:
    """Complete regimen interaction report."""

    drugs: list[str]
    interactions: list[Interaction]
    n_contraindicated: int = 0
    n_major: int = 0
    n_moderate: int = 0
    n_minor: int = 0
    qt_risk: str = "low"
    nephrotoxicity_risk: str = "low"
    hepatotoxicity_risk: str = "low"
    summary: str = ""
    recommendations: list[str] = field(default_factory=list)
    safe_to_use: bool = True


# =============================================================================
# Drug Database
# =============================================================================

DRUG_DATABASE: dict[str, DrugInfo] = {
    # HIV Protease Inhibitors (strong CYP3A4 inhibitors)
    "ritonavir": DrugInfo(
        name="Ritonavir",
        generic_name="ritonavir",
        category=DrugCategory.HIV_BOOSTER,
        cyp3a4_inhibitor=True,
        cyp2d6_inhibitor=True,
        pgp_inhibitor=True,
        aliases=["RTV", "Norvir"],
    ),
    "lopinavir": DrugInfo(
        name="Lopinavir",
        generic_name="lopinavir",
        category=DrugCategory.HIV_PI,
        cyp3a4_substrate=True,
        cyp3a4_inhibitor=True,
        qt_prolongation=True,
        aliases=["LPV", "Kaletra"],
    ),
    "darunavir": DrugInfo(
        name="Darunavir",
        generic_name="darunavir",
        category=DrugCategory.HIV_PI,
        cyp3a4_substrate=True,
        cyp3a4_inhibitor=True,
        aliases=["DRV", "Prezista"],
    ),
    "atazanavir": DrugInfo(
        name="Atazanavir",
        generic_name="atazanavir",
        category=DrugCategory.HIV_PI,
        cyp3a4_substrate=True,
        cyp3a4_inhibitor=True,
        qt_prolongation=True,
        aliases=["ATV", "Reyataz"],
    ),
    # HIV NRTIs
    "tenofovir": DrugInfo(
        name="Tenofovir",
        generic_name="tenofovir",
        category=DrugCategory.HIV_NRTI,
        nephrotoxic=True,
        aliases=["TDF", "Viread", "tenofovir_df"],
    ),
    "lamivudine": DrugInfo(
        name="Lamivudine",
        generic_name="lamivudine",
        category=DrugCategory.HIV_NRTI,
        aliases=["3TC", "Epivir"],
    ),
    "abacavir": DrugInfo(
        name="Abacavir",
        generic_name="abacavir",
        category=DrugCategory.HIV_NRTI,
        aliases=["ABC", "Ziagen"],
    ),
    "zidovudine": DrugInfo(
        name="Zidovudine",
        generic_name="zidovudine",
        category=DrugCategory.HIV_NRTI,
        myelosuppressive=True,
        aliases=["AZT", "ZDV", "Retrovir"],
    ),
    "emtricitabine": DrugInfo(
        name="Emtricitabine",
        generic_name="emtricitabine",
        category=DrugCategory.HIV_NRTI,
        aliases=["FTC", "Emtriva"],
    ),
    # HIV NNRTIs
    "efavirenz": DrugInfo(
        name="Efavirenz",
        generic_name="efavirenz",
        category=DrugCategory.HIV_NNRTI,
        cyp3a4_substrate=True,
        cyp3a4_inducer=True,
        cyp2d6_inhibitor=True,
        aliases=["EFV", "Sustiva", "Stocrin"],
    ),
    "nevirapine": DrugInfo(
        name="Nevirapine",
        generic_name="nevirapine",
        category=DrugCategory.HIV_NNRTI,
        cyp3a4_substrate=True,
        cyp3a4_inducer=True,
        hepatotoxic=True,
        aliases=["NVP", "Viramune"],
    ),
    "rilpivirine": DrugInfo(
        name="Rilpivirine",
        generic_name="rilpivirine",
        category=DrugCategory.HIV_NNRTI,
        cyp3a4_substrate=True,
        qt_prolongation=True,
        aliases=["RPV", "Edurant"],
    ),
    "doravirine": DrugInfo(
        name="Doravirine",
        generic_name="doravirine",
        category=DrugCategory.HIV_NNRTI,
        cyp3a4_substrate=True,
        aliases=["DOR", "Pifeltro"],
    ),
    # HIV Integrase Inhibitors
    "dolutegravir": DrugInfo(
        name="Dolutegravir",
        generic_name="dolutegravir",
        category=DrugCategory.HIV_INI,
        aliases=["DTG", "Tivicay"],
    ),
    "raltegravir": DrugInfo(
        name="Raltegravir",
        generic_name="raltegravir",
        category=DrugCategory.HIV_INI,
        aliases=["RAL", "Isentress"],
    ),
    "bictegravir": DrugInfo(
        name="Bictegravir",
        generic_name="bictegravir",
        category=DrugCategory.HIV_INI,
        aliases=["BIC", "Biktarvy"],
    ),
    "cabotegravir": DrugInfo(
        name="Cabotegravir",
        generic_name="cabotegravir",
        category=DrugCategory.HIV_INI,
        aliases=["CAB", "Vocabria"],
    ),
    # Azole Antifungals (CYP3A4 inhibitors)
    "fluconazole": DrugInfo(
        name="Fluconazole",
        generic_name="fluconazole",
        category=DrugCategory.AZOLE_ANTIFUNGAL,
        cyp3a4_inhibitor=True,
        cyp2c19_inhibitor=True,
        qt_prolongation=True,
        aliases=["Diflucan"],
    ),
    "itraconazole": DrugInfo(
        name="Itraconazole",
        generic_name="itraconazole",
        category=DrugCategory.AZOLE_ANTIFUNGAL,
        cyp3a4_substrate=True,
        cyp3a4_inhibitor=True,
        pgp_inhibitor=True,
        qt_prolongation=True,
        aliases=["Sporanox"],
    ),
    "voriconazole": DrugInfo(
        name="Voriconazole",
        generic_name="voriconazole",
        category=DrugCategory.AZOLE_ANTIFUNGAL,
        cyp3a4_substrate=True,
        cyp3a4_inhibitor=True,
        cyp2c19_inhibitor=True,
        qt_prolongation=True,
        hepatotoxic=True,
        aliases=["Vfend"],
    ),
    "posaconazole": DrugInfo(
        name="Posaconazole",
        generic_name="posaconazole",
        category=DrugCategory.AZOLE_ANTIFUNGAL,
        cyp3a4_inhibitor=True,
        pgp_inhibitor=True,
        qt_prolongation=True,
        aliases=["Noxafil"],
    ),
    "isavuconazole": DrugInfo(
        name="Isavuconazole",
        generic_name="isavuconazole",
        category=DrugCategory.AZOLE_ANTIFUNGAL,
        cyp3a4_substrate=True,
        cyp3a4_inhibitor=True,
        aliases=["Cresemba"],
    ),
    # Echinocandins
    "caspofungin": DrugInfo(
        name="Caspofungin",
        generic_name="caspofungin",
        category=DrugCategory.ECHINOCANDIN,
        hepatotoxic=True,
        aliases=["Cancidas"],
    ),
    "micafungin": DrugInfo(
        name="Micafungin",
        generic_name="micafungin",
        category=DrugCategory.ECHINOCANDIN,
        hepatotoxic=True,
        aliases=["Mycamine"],
    ),
    "anidulafungin": DrugInfo(
        name="Anidulafungin",
        generic_name="anidulafungin",
        category=DrugCategory.ECHINOCANDIN,
        aliases=["Eraxis"],
    ),
    # Polyenes
    "amphotericin_b": DrugInfo(
        name="Amphotericin B",
        generic_name="amphotericin_b",
        category=DrugCategory.POLYENE,
        nephrotoxic=True,
        aliases=["AmB", "Fungizone", "Abelcet", "AmBisome"],
    ),
    # Aminoglycosides
    "gentamicin": DrugInfo(
        name="Gentamicin",
        generic_name="gentamicin",
        category=DrugCategory.AMINOGLYCOSIDE,
        nephrotoxic=True,
        aliases=["Garamycin"],
    ),
    "amikacin": DrugInfo(
        name="Amikacin",
        generic_name="amikacin",
        category=DrugCategory.AMINOGLYCOSIDE,
        nephrotoxic=True,
        aliases=["Amikin"],
    ),
    "tobramycin": DrugInfo(
        name="Tobramycin",
        generic_name="tobramycin",
        category=DrugCategory.AMINOGLYCOSIDE,
        nephrotoxic=True,
        aliases=["Tobrex", "TOBI"],
    ),
    "streptomycin": DrugInfo(
        name="Streptomycin",
        generic_name="streptomycin",
        category=DrugCategory.AMINOGLYCOSIDE,
        nephrotoxic=True,
    ),
    # Glycopeptides
    "vancomycin": DrugInfo(
        name="Vancomycin",
        generic_name="vancomycin",
        category=DrugCategory.GLYCOPEPTIDE,
        nephrotoxic=True,
        aliases=["Vancocin"],
    ),
    "teicoplanin": DrugInfo(
        name="Teicoplanin",
        generic_name="teicoplanin",
        category=DrugCategory.GLYCOPEPTIDE,
        nephrotoxic=True,
        aliases=["Targocid"],
    ),
    # Fluoroquinolones
    "ciprofloxacin": DrugInfo(
        name="Ciprofloxacin",
        generic_name="ciprofloxacin",
        category=DrugCategory.FLUOROQUINOLONE,
        cyp3a4_inhibitor=True,
        qt_prolongation=True,
        aliases=["Cipro"],
    ),
    "levofloxacin": DrugInfo(
        name="Levofloxacin",
        generic_name="levofloxacin",
        category=DrugCategory.FLUOROQUINOLONE,
        qt_prolongation=True,
        aliases=["Levaquin"],
    ),
    "moxifloxacin": DrugInfo(
        name="Moxifloxacin",
        generic_name="moxifloxacin",
        category=DrugCategory.FLUOROQUINOLONE,
        qt_prolongation=True,
        aliases=["Avelox"],
    ),
    # Macrolides
    "azithromycin": DrugInfo(
        name="Azithromycin",
        generic_name="azithromycin",
        category=DrugCategory.MACROLIDE,
        qt_prolongation=True,
        aliases=["Zithromax", "Z-pack"],
    ),
    "clarithromycin": DrugInfo(
        name="Clarithromycin",
        generic_name="clarithromycin",
        category=DrugCategory.MACROLIDE,
        cyp3a4_inhibitor=True,
        qt_prolongation=True,
        aliases=["Biaxin"],
    ),
    "erythromycin": DrugInfo(
        name="Erythromycin",
        generic_name="erythromycin",
        category=DrugCategory.MACROLIDE,
        cyp3a4_inhibitor=True,
        qt_prolongation=True,
        aliases=["E-Mycin", "Ery-Tab"],
    ),
    # Polymyxins
    "colistin": DrugInfo(
        name="Colistin",
        generic_name="colistin",
        category=DrugCategory.POLYMYXIN,
        nephrotoxic=True,
        aliases=["polymyxin_e", "Coly-Mycin"],
    ),
    "polymyxin_b": DrugInfo(
        name="Polymyxin B",
        generic_name="polymyxin_b",
        category=DrugCategory.POLYMYXIN,
        nephrotoxic=True,
    ),
    # Oxazolidinones
    "linezolid": DrugInfo(
        name="Linezolid",
        generic_name="linezolid",
        category=DrugCategory.OXAZOLIDINONE,
        myelosuppressive=True,
        aliases=["Zyvox"],
    ),
    "tedizolid": DrugInfo(
        name="Tedizolid",
        generic_name="tedizolid",
        category=DrugCategory.OXAZOLIDINONE,
        aliases=["Sivextro"],
    ),
    # Beta-lactams
    "meropenem": DrugInfo(
        name="Meropenem",
        generic_name="meropenem",
        category=DrugCategory.BETA_LACTAM,
        aliases=["Merrem"],
    ),
    "imipenem": DrugInfo(
        name="Imipenem",
        generic_name="imipenem",
        category=DrugCategory.BETA_LACTAM,
        aliases=["Primaxin"],
    ),
    "piperacillin_tazobactam": DrugInfo(
        name="Piperacillin-Tazobactam",
        generic_name="piperacillin_tazobactam",
        category=DrugCategory.BETA_LACTAM,
        nephrotoxic=True,
        aliases=["Zosyn", "pip_tazo"],
    ),
    "ceftriaxone": DrugInfo(
        name="Ceftriaxone",
        generic_name="ceftriaxone",
        category=DrugCategory.BETA_LACTAM,
        aliases=["Rocephin"],
    ),
    "cefepime": DrugInfo(
        name="Cefepime",
        generic_name="cefepime",
        category=DrugCategory.BETA_LACTAM,
        aliases=["Maxipime"],
    ),
    "ceftazidime": DrugInfo(
        name="Ceftazidime",
        generic_name="ceftazidime",
        category=DrugCategory.BETA_LACTAM,
        aliases=["Fortaz", "Tazicef"],
    ),
    "ceftazidime_avibactam": DrugInfo(
        name="Ceftazidime-Avibactam",
        generic_name="ceftazidime_avibactam",
        category=DrugCategory.BETA_LACTAM,
        aliases=["Avycaz"],
    ),
    "ceftolozane_tazobactam": DrugInfo(
        name="Ceftolozane-Tazobactam",
        generic_name="ceftolozane_tazobactam",
        category=DrugCategory.BETA_LACTAM,
        aliases=["Zerbaxa"],
    ),
    "cefiderocol": DrugInfo(
        name="Cefiderocol",
        generic_name="cefiderocol",
        category=DrugCategory.BETA_LACTAM,
        aliases=["Fetroja"],
    ),
    # Tetracyclines
    "doxycycline": DrugInfo(
        name="Doxycycline",
        generic_name="doxycycline",
        category=DrugCategory.TETRACYCLINE,
        aliases=["Vibramycin"],
    ),
    "tigecycline": DrugInfo(
        name="Tigecycline",
        generic_name="tigecycline",
        category=DrugCategory.TETRACYCLINE,
        aliases=["Tygacil"],
    ),
    "eravacycline": DrugInfo(
        name="Eravacycline",
        generic_name="eravacycline",
        category=DrugCategory.TETRACYCLINE,
        aliases=["Xerava"],
    ),
    # TB drugs
    "rifampin": DrugInfo(
        name="Rifampin",
        generic_name="rifampin",
        category=DrugCategory.ANTIMALARIAL,  # Also TB
        cyp3a4_inducer=True,
        pgp_inducer=True,
        hepatotoxic=True,
        aliases=["rifampicin", "Rifadin"],
    ),
    "isoniazid": DrugInfo(
        name="Isoniazid",
        generic_name="isoniazid",
        category=DrugCategory.ANTIMALARIAL,
        hepatotoxic=True,
        aliases=["INH", "Nydrazid"],
    ),
    "pyrazinamide": DrugInfo(
        name="Pyrazinamide",
        generic_name="pyrazinamide",
        category=DrugCategory.ANTIMALARIAL,
        hepatotoxic=True,
        aliases=["PZA"],
    ),
    "ethambutol": DrugInfo(
        name="Ethambutol",
        generic_name="ethambutol",
        category=DrugCategory.ANTIMALARIAL,
        aliases=["EMB", "Myambutol"],
    ),
    "bedaquiline": DrugInfo(
        name="Bedaquiline",
        generic_name="bedaquiline",
        category=DrugCategory.ANTIMALARIAL,
        cyp3a4_substrate=True,
        qt_prolongation=True,
        aliases=["Sirturo"],
    ),
    # HCV DAAs
    "sofosbuvir": DrugInfo(
        name="Sofosbuvir",
        generic_name="sofosbuvir",
        category=DrugCategory.HCV_DAA,
        pgp_substrate=True,
        aliases=["Sovaldi"],
    ),
    "ledipasvir": DrugInfo(
        name="Ledipasvir",
        generic_name="ledipasvir",
        category=DrugCategory.HCV_DAA,
        aliases=["Harvoni"],
    ),
    "velpatasvir": DrugInfo(
        name="Velpatasvir",
        generic_name="velpatasvir",
        category=DrugCategory.HCV_DAA,
        cyp3a4_substrate=True,
        aliases=["Epclusa"],
    ),
    "glecaprevir": DrugInfo(
        name="Glecaprevir",
        generic_name="glecaprevir",
        category=DrugCategory.HCV_DAA,
        cyp3a4_substrate=True,
        pgp_substrate=True,
        aliases=["Mavyret"],
    ),
    "pibrentasvir": DrugInfo(
        name="Pibrentasvir",
        generic_name="pibrentasvir",
        category=DrugCategory.HCV_DAA,
        pgp_substrate=True,
        aliases=["Mavyret"],
    ),
    # Antimalarials
    "artemether_lumefantrine": DrugInfo(
        name="Artemether-Lumefantrine",
        generic_name="artemether_lumefantrine",
        category=DrugCategory.ANTIMALARIAL,
        cyp3a4_substrate=True,
        qt_prolongation=True,
        aliases=["Coartem", "AL"],
    ),
    "chloroquine": DrugInfo(
        name="Chloroquine",
        generic_name="chloroquine",
        category=DrugCategory.ANTIMALARIAL,
        qt_prolongation=True,
        aliases=["Aralen"],
    ),
    "mefloquine": DrugInfo(
        name="Mefloquine",
        generic_name="mefloquine",
        category=DrugCategory.ANTIMALARIAL,
        qt_prolongation=True,
        aliases=["Lariam"],
    ),
    "atovaquone_proguanil": DrugInfo(
        name="Atovaquone-Proguanil",
        generic_name="atovaquone_proguanil",
        category=DrugCategory.ANTIMALARIAL,
        aliases=["Malarone"],
    ),
    # Statins (common DDI concern)
    "simvastatin": DrugInfo(
        name="Simvastatin",
        generic_name="simvastatin",
        category=DrugCategory.STATIN,
        cyp3a4_substrate=True,
        aliases=["Zocor"],
    ),
    "atorvastatin": DrugInfo(
        name="Atorvastatin",
        generic_name="atorvastatin",
        category=DrugCategory.STATIN,
        cyp3a4_substrate=True,
        aliases=["Lipitor"],
    ),
    "lovastatin": DrugInfo(
        name="Lovastatin",
        generic_name="lovastatin",
        category=DrugCategory.STATIN,
        cyp3a4_substrate=True,
        aliases=["Mevacor"],
    ),
    "rosuvastatin": DrugInfo(
        name="Rosuvastatin",
        generic_name="rosuvastatin",
        category=DrugCategory.STATIN,
        aliases=["Crestor"],
    ),
    "pravastatin": DrugInfo(
        name="Pravastatin",
        generic_name="pravastatin",
        category=DrugCategory.STATIN,
        aliases=["Pravachol"],
    ),
    # Cancer TKIs
    "imatinib": DrugInfo(
        name="Imatinib",
        generic_name="imatinib",
        category=DrugCategory.CANCER_TKI,
        cyp3a4_substrate=True,
        cyp3a4_inhibitor=True,
        hepatotoxic=True,
        aliases=["Gleevec"],
    ),
    "erlotinib": DrugInfo(
        name="Erlotinib",
        generic_name="erlotinib",
        category=DrugCategory.CANCER_TKI,
        cyp3a4_substrate=True,
        aliases=["Tarceva"],
    ),
    "osimertinib": DrugInfo(
        name="Osimertinib",
        generic_name="osimertinib",
        category=DrugCategory.CANCER_TKI,
        cyp3a4_substrate=True,
        qt_prolongation=True,
        aliases=["Tagrisso"],
    ),
}


# =============================================================================
# Specific Known Interactions
# =============================================================================

KNOWN_INTERACTIONS: list[Interaction] = [
    # Ritonavir + Statins (contraindicated)
    Interaction(
        drug1="ritonavir",
        drug2="simvastatin",
        severity=InteractionSeverity.CONTRAINDICATED,
        mechanism=InteractionMechanism.CYP3A4_INHIBITION,
        description="Ritonavir strongly inhibits CYP3A4, dramatically increasing simvastatin levels",
        clinical_effect="Risk of rhabdomyolysis increased 10-40 fold",
        recommendation="Do not coadminister. Use pravastatin or rosuvastatin instead.",
        alternatives=["pravastatin", "rosuvastatin"],
    ),
    Interaction(
        drug1="ritonavir",
        drug2="lovastatin",
        severity=InteractionSeverity.CONTRAINDICATED,
        mechanism=InteractionMechanism.CYP3A4_INHIBITION,
        description="Ritonavir strongly inhibits CYP3A4, dramatically increasing lovastatin levels",
        clinical_effect="Risk of rhabdomyolysis increased 10-40 fold",
        recommendation="Do not coadminister. Use pravastatin or rosuvastatin instead.",
        alternatives=["pravastatin", "rosuvastatin"],
    ),
    Interaction(
        drug1="ritonavir",
        drug2="atorvastatin",
        severity=InteractionSeverity.MAJOR,
        mechanism=InteractionMechanism.CYP3A4_INHIBITION,
        description="Ritonavir increases atorvastatin levels 3-9 fold",
        clinical_effect="Increased risk of myopathy and rhabdomyolysis",
        recommendation="Start with lowest atorvastatin dose (10mg). Consider pravastatin or rosuvastatin.",
        alternatives=["pravastatin", "rosuvastatin"],
        monitoring=["creatine kinase", "muscle symptoms"],
    ),
    # Rifampin interactions (strong CYP3A4 inducer)
    Interaction(
        drug1="rifampin",
        drug2="ritonavir",
        severity=InteractionSeverity.CONTRAINDICATED,
        mechanism=InteractionMechanism.CYP3A4_INDUCTION,
        description="Rifampin dramatically reduces ritonavir levels through CYP3A4 induction",
        clinical_effect="HIV treatment failure, resistance development",
        recommendation="Do not coadminister. Use rifabutin instead if needed.",
        alternatives=["rifabutin"],
    ),
    Interaction(
        drug1="rifampin",
        drug2="lopinavir",
        severity=InteractionSeverity.CONTRAINDICATED,
        mechanism=InteractionMechanism.CYP3A4_INDUCTION,
        description="Rifampin reduces lopinavir levels by 75%",
        clinical_effect="HIV treatment failure, resistance development",
        recommendation="Do not coadminister. Use rifabutin instead.",
        alternatives=["rifabutin"],
    ),
    Interaction(
        drug1="rifampin",
        drug2="darunavir",
        severity=InteractionSeverity.CONTRAINDICATED,
        mechanism=InteractionMechanism.CYP3A4_INDUCTION,
        description="Rifampin reduces darunavir levels by >80%",
        clinical_effect="HIV treatment failure",
        recommendation="Do not coadminister. Use rifabutin instead.",
        alternatives=["rifabutin"],
    ),
    Interaction(
        drug1="rifampin",
        drug2="dolutegravir",
        severity=InteractionSeverity.MAJOR,
        mechanism=InteractionMechanism.CYP3A4_INDUCTION,
        description="Rifampin reduces dolutegravir levels by 50-80%",
        clinical_effect="May lead to treatment failure",
        recommendation="Increase dolutegravir to 50mg twice daily during rifampin coadministration.",
        monitoring=["HIV viral load"],
    ),
    Interaction(
        drug1="rifampin",
        drug2="voriconazole",
        severity=InteractionSeverity.CONTRAINDICATED,
        mechanism=InteractionMechanism.CYP3A4_INDUCTION,
        description="Rifampin reduces voriconazole levels by >90%",
        clinical_effect="Antifungal treatment failure",
        recommendation="Do not coadminister. Use alternative antifungal.",
        alternatives=["isavuconazole", "caspofungin"],
    ),
    Interaction(
        drug1="rifampin",
        drug2="itraconazole",
        severity=InteractionSeverity.CONTRAINDICATED,
        mechanism=InteractionMechanism.CYP3A4_INDUCTION,
        description="Rifampin reduces itraconazole levels by >90%",
        clinical_effect="Antifungal treatment failure",
        recommendation="Do not coadminister.",
        alternatives=["caspofungin"],
    ),
    # Nephrotoxicity synergy
    Interaction(
        drug1="vancomycin",
        drug2="gentamicin",
        severity=InteractionSeverity.MAJOR,
        mechanism=InteractionMechanism.NEPHROTOXICITY_SYNERGY,
        description="Both drugs are nephrotoxic with synergistic effect",
        clinical_effect="Increased risk of acute kidney injury",
        recommendation="Avoid combination if possible. If necessary, monitor renal function closely.",
        monitoring=["serum creatinine", "BUN", "urine output", "drug levels"],
    ),
    Interaction(
        drug1="vancomycin",
        drug2="piperacillin_tazobactam",
        severity=InteractionSeverity.MAJOR,
        mechanism=InteractionMechanism.NEPHROTOXICITY_SYNERGY,
        description="Combination increases AKI risk vs vancomycin alone",
        clinical_effect="2-3 fold increased risk of acute kidney injury",
        recommendation="Consider cefepime instead of piperacillin-tazobactam if possible.",
        monitoring=["serum creatinine", "BUN"],
        alternatives=["cefepime"],
    ),
    Interaction(
        drug1="amphotericin_b",
        drug2="gentamicin",
        severity=InteractionSeverity.MAJOR,
        mechanism=InteractionMechanism.NEPHROTOXICITY_SYNERGY,
        description="Both drugs are highly nephrotoxic",
        clinical_effect="Severe acute kidney injury risk",
        recommendation="Avoid combination. Use liposomal amphotericin if possible.",
        monitoring=["serum creatinine", "potassium", "magnesium"],
        alternatives=["liposomal amphotericin_b"],
    ),
    Interaction(
        drug1="tenofovir",
        drug2="gentamicin",
        severity=InteractionSeverity.MAJOR,
        mechanism=InteractionMechanism.NEPHROTOXICITY_SYNERGY,
        description="Both drugs can cause renal tubular toxicity",
        clinical_effect="Increased risk of Fanconi syndrome and AKI",
        recommendation="Avoid combination. If necessary, monitor renal function closely.",
        monitoring=["serum creatinine", "phosphate", "glucose"],
    ),
    Interaction(
        drug1="colistin",
        drug2="vancomycin",
        severity=InteractionSeverity.MAJOR,
        mechanism=InteractionMechanism.NEPHROTOXICITY_SYNERGY,
        description="Both drugs are nephrotoxic",
        clinical_effect="High risk of acute kidney injury",
        recommendation="Avoid if possible. Monitor renal function daily.",
        monitoring=["serum creatinine", "BUN", "urine output"],
    ),
    # QT prolongation combinations
    Interaction(
        drug1="moxifloxacin",
        drug2="azithromycin",
        severity=InteractionSeverity.MAJOR,
        mechanism=InteractionMechanism.QT_PROLONGATION,
        description="Both drugs prolong QT interval",
        clinical_effect="Risk of torsades de pointes",
        recommendation="Avoid combination. Use alternative antibiotic.",
        monitoring=["ECG", "QTc interval", "potassium", "magnesium"],
    ),
    Interaction(
        drug1="moxifloxacin",
        drug2="fluconazole",
        severity=InteractionSeverity.MAJOR,
        mechanism=InteractionMechanism.QT_PROLONGATION,
        description="Both drugs prolong QT interval",
        clinical_effect="Risk of torsades de pointes",
        recommendation="Avoid combination if possible. Monitor ECG.",
        monitoring=["ECG", "QTc interval"],
    ),
    Interaction(
        drug1="lopinavir",
        drug2="bedaquiline",
        severity=InteractionSeverity.MAJOR,
        mechanism=InteractionMechanism.QT_PROLONGATION,
        description="Both drugs prolong QT interval; lopinavir also increases bedaquiline levels",
        clinical_effect="Risk of cardiac arrhythmias",
        recommendation="Use with caution. Monitor ECG frequently.",
        monitoring=["ECG", "QTc interval"],
    ),
    Interaction(
        drug1="chloroquine",
        drug2="azithromycin",
        severity=InteractionSeverity.MAJOR,
        mechanism=InteractionMechanism.QT_PROLONGATION,
        description="Both drugs prolong QT interval",
        clinical_effect="Risk of torsades de pointes",
        recommendation="Avoid combination. FDA warning against this combination.",
        monitoring=["ECG", "QTc interval"],
    ),
    # Linezolid interactions
    Interaction(
        drug1="linezolid",
        drug2="zidovudine",
        severity=InteractionSeverity.MAJOR,
        mechanism=InteractionMechanism.BONE_MARROW_SUPPRESSION,
        description="Both drugs cause myelosuppression",
        clinical_effect="Increased risk of anemia and pancytopenia",
        recommendation="Avoid combination. Use alternative agents.",
        monitoring=["CBC", "hemoglobin"],
    ),
    # Azole-PI combinations
    Interaction(
        drug1="voriconazole",
        drug2="efavirenz",
        severity=InteractionSeverity.CONTRAINDICATED,
        mechanism=InteractionMechanism.CYP3A4_INDUCTION,
        description="Efavirenz reduces voriconazole levels while voriconazole increases efavirenz levels",
        clinical_effect="Treatment failure for both drugs possible",
        recommendation="Do not coadminister at standard doses.",
        alternatives=["fluconazole", "isavuconazole"],
    ),
    # HCV DAA interactions
    Interaction(
        drug1="sofosbuvir",
        drug2="rifampin",
        severity=InteractionSeverity.CONTRAINDICATED,
        mechanism=InteractionMechanism.PGLYCOPROTEIN_INDUCTION,
        description="Rifampin induces P-gp, dramatically reducing sofosbuvir levels",
        clinical_effect="HCV treatment failure",
        recommendation="Do not coadminister. Complete HCV treatment before TB treatment if possible.",
    ),
    Interaction(
        drug1="glecaprevir",
        drug2="atazanavir",
        severity=InteractionSeverity.CONTRAINDICATED,
        mechanism=InteractionMechanism.CYP3A4_INHIBITION,
        description="Atazanavir significantly increases glecaprevir levels",
        clinical_effect="Increased hepatotoxicity risk",
        recommendation="Do not coadminister.",
    ),
]


class DrugInteractionChecker:
    """Drug-drug interaction checker for antimicrobial therapy."""

    def __init__(self):
        """Initialize the interaction checker."""
        self.drug_database = DRUG_DATABASE
        self.known_interactions = KNOWN_INTERACTIONS
        self._build_interaction_index()

    def _build_interaction_index(self) -> None:
        """Build index for fast interaction lookup."""
        self._interaction_index: dict[tuple[str, str], Interaction] = {}
        for interaction in self.known_interactions:
            # Index both directions
            key1 = (interaction.drug1.lower(), interaction.drug2.lower())
            key2 = (interaction.drug2.lower(), interaction.drug1.lower())
            self._interaction_index[key1] = interaction
            self._interaction_index[key2] = interaction

    def _normalize_drug_name(self, drug: str) -> str | None:
        """Normalize drug name to generic name."""
        drug_lower = drug.lower().replace("-", "_").replace(" ", "_")

        # Check if it's already a known drug
        if drug_lower in self.drug_database:
            return drug_lower

        # Check aliases
        for generic, info in self.drug_database.items():
            if drug_lower in [a.lower().replace("-", "_") for a in info.aliases]:
                return generic
            if drug_lower == info.name.lower().replace("-", "_"):
                return generic

        return None

    def get_drug_info(self, drug: str) -> DrugInfo | None:
        """Get drug information."""
        normalized = self._normalize_drug_name(drug)
        if normalized:
            return self.drug_database.get(normalized)
        return None

    def check_interaction(self, drug1: str, drug2: str) -> Interaction | None:
        """Check for interaction between two drugs.

        Args:
            drug1: First drug name
            drug2: Second drug name

        Returns:
            Interaction object if interaction exists, None otherwise
        """
        norm1 = self._normalize_drug_name(drug1)
        norm2 = self._normalize_drug_name(drug2)

        if not norm1 or not norm2:
            return None

        if norm1 == norm2:
            return None

        # Check known interactions
        key = (norm1, norm2)
        if key in self._interaction_index:
            return self._interaction_index[key]

        # Check for mechanism-based interactions
        return self._infer_interaction(norm1, norm2)

    def _infer_interaction(self, drug1: str, drug2: str) -> Interaction | None:
        """Infer interaction based on drug properties."""
        info1 = self.drug_database.get(drug1)
        info2 = self.drug_database.get(drug2)

        if not info1 or not info2:
            return None

        # CYP3A4 inhibitor + substrate
        if info1.cyp3a4_inhibitor and info2.cyp3a4_substrate:
            return Interaction(
                drug1=drug1,
                drug2=drug2,
                severity=InteractionSeverity.MODERATE,
                mechanism=InteractionMechanism.CYP3A4_INHIBITION,
                description=f"{info1.name} inhibits CYP3A4, potentially increasing {info2.name} levels",
                clinical_effect=f"Increased {info2.name} exposure may occur",
                recommendation=f"Monitor for increased {info2.name} effects. Consider dose reduction.",
                monitoring=["therapeutic drug monitoring if available"],
            )

        if info2.cyp3a4_inhibitor and info1.cyp3a4_substrate:
            return Interaction(
                drug1=drug1,
                drug2=drug2,
                severity=InteractionSeverity.MODERATE,
                mechanism=InteractionMechanism.CYP3A4_INHIBITION,
                description=f"{info2.name} inhibits CYP3A4, potentially increasing {info1.name} levels",
                clinical_effect=f"Increased {info1.name} exposure may occur",
                recommendation=f"Monitor for increased {info1.name} effects. Consider dose reduction.",
                monitoring=["therapeutic drug monitoring if available"],
            )

        # CYP3A4 inducer + substrate
        if info1.cyp3a4_inducer and info2.cyp3a4_substrate:
            return Interaction(
                drug1=drug1,
                drug2=drug2,
                severity=InteractionSeverity.MAJOR,
                mechanism=InteractionMechanism.CYP3A4_INDUCTION,
                description=f"{info1.name} induces CYP3A4, potentially decreasing {info2.name} levels",
                clinical_effect=f"Reduced {info2.name} efficacy may occur",
                recommendation=f"Consider alternative to {info1.name} or increase {info2.name} dose.",
            )

        if info2.cyp3a4_inducer and info1.cyp3a4_substrate:
            return Interaction(
                drug1=drug1,
                drug2=drug2,
                severity=InteractionSeverity.MAJOR,
                mechanism=InteractionMechanism.CYP3A4_INDUCTION,
                description=f"{info2.name} induces CYP3A4, potentially decreasing {info1.name} levels",
                clinical_effect=f"Reduced {info1.name} efficacy may occur",
                recommendation=f"Consider alternative to {info2.name} or increase {info1.name} dose.",
            )

        # QT prolongation
        if info1.qt_prolongation and info2.qt_prolongation:
            return Interaction(
                drug1=drug1,
                drug2=drug2,
                severity=InteractionSeverity.MAJOR,
                mechanism=InteractionMechanism.QT_PROLONGATION,
                description="Both drugs prolong QT interval",
                clinical_effect="Increased risk of cardiac arrhythmias including torsades de pointes",
                recommendation="Avoid combination if possible. Monitor ECG and electrolytes.",
                monitoring=["ECG", "QTc interval", "potassium", "magnesium"],
            )

        # Nephrotoxicity
        if info1.nephrotoxic and info2.nephrotoxic:
            return Interaction(
                drug1=drug1,
                drug2=drug2,
                severity=InteractionSeverity.MAJOR,
                mechanism=InteractionMechanism.NEPHROTOXICITY_SYNERGY,
                description="Both drugs are nephrotoxic",
                clinical_effect="Increased risk of acute kidney injury",
                recommendation="Avoid combination if possible. Monitor renal function closely.",
                monitoring=["serum creatinine", "BUN", "urine output"],
            )

        # Hepatotoxicity
        if info1.hepatotoxic and info2.hepatotoxic:
            return Interaction(
                drug1=drug1,
                drug2=drug2,
                severity=InteractionSeverity.MODERATE,
                mechanism=InteractionMechanism.HEPATOTOXICITY_SYNERGY,
                description="Both drugs are hepatotoxic",
                clinical_effect="Increased risk of liver injury",
                recommendation="Monitor liver function tests closely.",
                monitoring=["ALT", "AST", "bilirubin"],
            )

        # Myelosuppression
        if info1.myelosuppressive and info2.myelosuppressive:
            return Interaction(
                drug1=drug1,
                drug2=drug2,
                severity=InteractionSeverity.MAJOR,
                mechanism=InteractionMechanism.BONE_MARROW_SUPPRESSION,
                description="Both drugs cause bone marrow suppression",
                clinical_effect="Increased risk of pancytopenia",
                recommendation="Monitor CBC closely. Consider alternative agents.",
                monitoring=["CBC", "platelet count"],
            )

        return None

    def check_regimen(self, drugs: list[str]) -> RegimenReport:
        """Check all interactions in a drug regimen.

        Args:
            drugs: List of drug names in the regimen

        Returns:
            Complete regimen interaction report
        """
        interactions = []
        normalized_drugs = []

        # Normalize all drug names
        for drug in drugs:
            normalized = self._normalize_drug_name(drug)
            if normalized:
                normalized_drugs.append(normalized)

        # Check all pairs
        for i, drug1 in enumerate(normalized_drugs):
            for drug2 in normalized_drugs[i + 1 :]:
                interaction = self.check_interaction(drug1, drug2)
                if interaction:
                    interactions.append(interaction)

        # Count by severity
        n_contraindicated = sum(1 for i in interactions if i.severity == InteractionSeverity.CONTRAINDICATED)
        n_major = sum(1 for i in interactions if i.severity == InteractionSeverity.MAJOR)
        n_moderate = sum(1 for i in interactions if i.severity == InteractionSeverity.MODERATE)
        n_minor = sum(1 for i in interactions if i.severity == InteractionSeverity.MINOR)

        # Assess overall risks
        qt_drugs = [
            d
            for d in normalized_drugs
            if self.drug_database.get(d, DrugInfo("", "", DrugCategory.BETA_LACTAM)).qt_prolongation
        ]
        nephro_drugs = [
            d
            for d in normalized_drugs
            if self.drug_database.get(d, DrugInfo("", "", DrugCategory.BETA_LACTAM)).nephrotoxic
        ]
        hepato_drugs = [
            d
            for d in normalized_drugs
            if self.drug_database.get(d, DrugInfo("", "", DrugCategory.BETA_LACTAM)).hepatotoxic
        ]

        qt_risk = "low"
        if len(qt_drugs) >= 3:
            qt_risk = "high"
        elif len(qt_drugs) >= 2:
            qt_risk = "moderate"

        nephro_risk = "low"
        if len(nephro_drugs) >= 3:
            nephro_risk = "high"
        elif len(nephro_drugs) >= 2:
            nephro_risk = "moderate"

        hepato_risk = "low"
        if len(hepato_drugs) >= 3:
            hepato_risk = "high"
        elif len(hepato_drugs) >= 2:
            hepato_risk = "moderate"

        # Generate recommendations
        recommendations = []
        if n_contraindicated > 0:
            recommendations.append("URGENT: Contraindicated drug combination(s) detected. Revise regimen immediately.")
        if n_major > 0:
            recommendations.append(
                f"CAUTION: {n_major} major interaction(s) detected. Consider alternatives or monitoring."
            )
        if qt_risk == "high":
            recommendations.append(
                "HIGH QT RISK: Multiple QT-prolonging drugs. Baseline ECG and electrolytes recommended."
            )
        if nephro_risk == "high":
            recommendations.append("HIGH NEPHROTOXICITY RISK: Monitor renal function daily.")
        if hepato_risk == "high":
            recommendations.append("HIGH HEPATOTOXICITY RISK: Monitor LFTs regularly.")

        # Generate summary
        if n_contraindicated > 0:
            summary = f"CRITICAL: {n_contraindicated} contraindicated interaction(s). Regimen revision required."
        elif n_major > 2:
            summary = f"HIGH RISK: {n_major} major interactions. Close monitoring required."
        elif n_major > 0:
            summary = f"MODERATE RISK: {n_major} major, {n_moderate} moderate interactions."
        elif n_moderate > 0:
            summary = f"LOW RISK: {n_moderate} moderate interactions. Standard monitoring."
        else:
            summary = "No significant interactions detected."

        return RegimenReport(
            drugs=normalized_drugs,
            interactions=interactions,
            n_contraindicated=n_contraindicated,
            n_major=n_major,
            n_moderate=n_moderate,
            n_minor=n_minor,
            qt_risk=qt_risk,
            nephrotoxicity_risk=nephro_risk,
            hepatotoxicity_risk=hepato_risk,
            summary=summary,
            recommendations=recommendations,
            safe_to_use=n_contraindicated == 0,
        )

    def get_alternatives(self, drug: str, category: DrugCategory | None = None) -> list[str]:
        """Get safer alternatives for a drug.

        Args:
            drug: Drug to find alternatives for
            category: Optional specific category to filter alternatives

        Returns:
            List of alternative drug names
        """
        normalized = self._normalize_drug_name(drug)
        if not normalized:
            return []

        drug_info = self.drug_database.get(normalized)
        if not drug_info:
            return []

        target_category = category or drug_info.category
        alternatives = []

        for name, info in self.drug_database.items():
            if name == normalized:
                continue
            if info.category != target_category:
                continue

            # Prefer drugs with fewer interaction concerns
            risk_score = 0
            if info.cyp3a4_inhibitor:
                risk_score += 1
            if info.cyp3a4_inducer:
                risk_score += 2
            if info.qt_prolongation:
                risk_score += 1
            if info.nephrotoxic:
                risk_score += 1
            if info.hepatotoxic:
                risk_score += 1

            alternatives.append((name, risk_score))

        # Sort by risk score
        alternatives.sort(key=lambda x: x[1])
        return [name for name, _ in alternatives[:5]]

    def list_drugs_by_category(self, category: DrugCategory) -> list[DrugInfo]:
        """List all drugs in a category."""
        return [info for info in self.drug_database.values() if info.category == category]

    def get_interaction_summary(self, drug: str) -> dict[str, Any]:
        """Get summary of all potential interactions for a drug.

        Args:
            drug: Drug name

        Returns:
            Dictionary with interaction summary
        """
        normalized = self._normalize_drug_name(drug)
        if not normalized:
            return {"error": f"Unknown drug: {drug}"}

        drug_info = self.drug_database.get(normalized)
        if not drug_info:
            return {"error": f"No information for: {drug}"}

        interactions = []
        for name in self.drug_database:
            if name != normalized:
                interaction = self.check_interaction(normalized, name)
                if interaction:
                    interactions.append(
                        {
                            "drug": name,
                            "severity": interaction.severity.value,
                            "mechanism": interaction.mechanism.value,
                        }
                    )

        # Sort by severity
        severity_order = {
            InteractionSeverity.CONTRAINDICATED.value: 0,
            InteractionSeverity.MAJOR.value: 1,
            InteractionSeverity.MODERATE.value: 2,
            InteractionSeverity.MINOR.value: 3,
        }
        interactions.sort(key=lambda x: severity_order.get(x["severity"], 4))

        return {
            "drug": normalized,
            "name": drug_info.name,
            "category": drug_info.category.value,
            "properties": {
                "cyp3a4_inhibitor": drug_info.cyp3a4_inhibitor,
                "cyp3a4_inducer": drug_info.cyp3a4_inducer,
                "cyp3a4_substrate": drug_info.cyp3a4_substrate,
                "qt_prolongation": drug_info.qt_prolongation,
                "nephrotoxic": drug_info.nephrotoxic,
                "hepatotoxic": drug_info.hepatotoxic,
                "myelosuppressive": drug_info.myelosuppressive,
            },
            "n_interactions": len(interactions),
            "interactions": interactions[:20],  # Top 20
        }
