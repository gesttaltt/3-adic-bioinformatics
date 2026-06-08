#!/usr/bin/env python3
"""
HIV Clinical Applications Pipeline

Implements actionable clinical tools based on research discoveries:
1. Vaccine Candidate Prioritization Pipeline
2. Multi-Drug Resistance (MDR) Early Warning Screener
3. Tat-Targeting Drug Opportunity Analysis
4. Unified Prediction Pipeline with P-adic Encoding
5. Clinical Decision Support Report Generator

These tools translate research findings into practical clinical applications.
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    print("ERROR: pandas required")
    sys.exit(1)

try:
    import pyarrow.parquet as pq

    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =============================================================================
# CONSTANTS
# =============================================================================

# MDR-associated mutations from Research 4
MDR_SIGNATURE_MUTATIONS = {
    "L63P": 0.795,  # 79.5% prevalence in MDR
    "L90M": 0.582,
    "A71V": 0.549,
    "M36I": 0.466,
    "L10I": 0.461,
    "I54V": 0.446,
    "M46I": 0.429,
    "L33F": 0.406,
    "I93L": 0.399,
    "V82A": 0.380,
}

# Top vaccine targets from Research 1
TOP_VACCINE_TARGETS = [
    {"epitope": "TPQDLNTML", "protein": "Gag", "stability_score": 173.66},
    {"epitope": "QVPLRPMTYK", "protein": "Nef", "stability_score": 133.87},
    {"epitope": "YFPDWQNYT", "protein": "Nef", "stability_score": 133.87},
    {"epitope": "YPLTFGWCF", "protein": "Nef", "stability_score": 133.87},
    {"epitope": "AAVDLSHFL", "protein": "Nef", "stability_score": 133.87},
]

# Druggable protein families
DRUGGABLE_FAMILIES = {
    "kinase": {"class": "enzyme", "druggability": "high"},
    "receptor": {"class": "membrane", "druggability": "high"},
    "protease": {"class": "enzyme", "druggability": "high"},
    "channel": {"class": "membrane", "druggability": "medium"},
    "transporter": {"class": "membrane", "druggability": "medium"},
    "phosphatase": {"class": "enzyme", "druggability": "medium"},
    "dehydrogenase": {"class": "enzyme", "druggability": "medium"},
    "ligase": {"class": "enzyme", "druggability": "low"},
}

# Amino acid encoding
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY-"
AA_GROUPS = {
    "hydrophobic_aliphatic": "AVILM",
    "hydrophobic_aromatic": "FWY",
    "polar_uncharged": "STNQ",
    "positive": "KRH",
    "negative": "DE",
    "special": "CGP",
}


# =============================================================================
# TOOL 1: VACCINE CANDIDATE PRIORITIZATION
# =============================================================================


class VaccinePrioritizer:
    """Prioritize vaccine candidates based on multiple criteria."""

    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.candidates = None
        self.hla_coverage = None

    def load_data(self):
        """Load vaccine candidate data."""
        # Load enhanced vaccine targets
        vaccine_path = self.results_dir / "integrated" / "vaccine_targets_with_stability.csv"
        if vaccine_path.exists():
            self.candidates = pd.read_csv(vaccine_path)
        else:
            # Fall back to original
            vaccine_path = self.results_dir / "integrated" / "vaccine_targets.csv"
            if vaccine_path.exists():
                self.candidates = pd.read_csv(vaccine_path)

        # Load HLA data if available
        hla_path = self.results_dir / "ctl_escape" / "hla_summary.csv"
        if hla_path.exists():
            self.hla_coverage = pd.read_csv(hla_path)

        return self.candidates is not None

    def compute_population_coverage(self, epitope_row) -> float:
        """Estimate population coverage based on HLA restrictions."""
        # Common HLA frequencies in global population

        n_hla = epitope_row.get("n_hla_restrictions", epitope_row.get("hla_count", 5))
        # Estimate coverage based on number of HLA restrictions
        # More HLAs = better population coverage
        base_coverage = min(0.95, 0.3 + 0.05 * n_hla)
        return base_coverage

    def compute_manufacturability_score(self, epitope: str) -> float:
        """Estimate peptide manufacturability."""
        if not isinstance(epitope, str):
            return 0.5

        score = 1.0

        # Penalize difficult amino acids
        difficult_aas = {"C": 0.1, "M": 0.05, "W": 0.05}
        for aa, penalty in difficult_aas.items():
            score -= epitope.count(aa) * penalty

        # Penalize very long or short peptides
        length = len(epitope)
        if length < 8 or length > 12:
            score -= 0.1

        # Penalize high hydrophobicity (solubility issues)
        hydrophobic = sum(1 for aa in epitope if aa in "AVILMFWY")
        if hydrophobic / max(length, 1) > 0.6:
            score -= 0.15

        return max(0, score)

    def prioritize(self, top_n: int = 50) -> pd.DataFrame:
        """Generate prioritized vaccine candidate list."""
        if self.candidates is None and not self.load_data():
            return pd.DataFrame()

        df = self.candidates.copy()

        # Compute additional scores
        df["population_coverage"] = df.apply(self.compute_population_coverage, axis=1)

        epitope_col = "epitope" if "epitope" in df.columns else "Epitope"
        df["manufacturability"] = df[epitope_col].apply(self.compute_manufacturability_score)

        # Get stability score
        if "stability_score" not in df.columns:
            df["stability_score"] = 50  # Default

        # Compute composite priority score
        df["priority_score"] = (
            0.4 * df["stability_score"] / df["stability_score"].max()
            + 0.3 * df["population_coverage"]
            + 0.3 * df["manufacturability"]
        )

        # Rank
        df = df.sort_values("priority_score", ascending=False)

        return df.head(top_n)

    def generate_report(self) -> dict:
        """Generate comprehensive vaccine prioritization report."""
        prioritized = self.prioritize(50)

        report = {
            "timestamp": datetime.now().isoformat(),
            "total_candidates": len(self.candidates) if self.candidates is not None else 0,
            "top_candidates": [],
            "protein_distribution": {},
            "recommendations": [],
        }

        epitope_col = "epitope" if "epitope" in prioritized.columns else "Epitope"
        protein_col = "protein" if "protein" in prioritized.columns else "Protein"

        for _, row in prioritized.head(10).iterrows():
            report["top_candidates"].append(
                {
                    "epitope": str(row[epitope_col]),
                    "protein": str(row[protein_col]),
                    "priority_score": float(row["priority_score"]),
                    "stability_score": float(row.get("stability_score", 0)),
                    "population_coverage": float(row["population_coverage"]),
                    "manufacturability": float(row["manufacturability"]),
                }
            )

        # Protein distribution
        if protein_col in prioritized.columns:
            report["protein_distribution"] = prioritized[protein_col].value_counts().head(10).to_dict()

        # Recommendations
        top = report["top_candidates"][0] if report["top_candidates"] else None
        if top:
            report["recommendations"].append(
                f"Primary candidate: {top['epitope']} from {top['protein']} (score: {top['priority_score']:.3f})"
            )
            report["recommendations"].append(
                "Recommended combination: Include epitopes from Gag, Nef, and Pol for broad coverage"
            )

        return report


# =============================================================================
# TOOL 2: MDR EARLY WARNING SCREENER
# =============================================================================


class MDRScreener:
    """Screen sequences for multi-drug resistance risk."""

    def __init__(self):
        self.signature_mutations = MDR_SIGNATURE_MUTATIONS
        self.model = None

    def parse_mutation_list(self, mut_string: str) -> list[str]:
        """Parse mutation string into list."""
        if not isinstance(mut_string, str):
            return []
        return [m.strip() for m in mut_string.split(",") if m.strip()]

    def compute_mdr_risk_score(self, mutations: list[str]) -> tuple[float, list[str]]:
        """
        Compute MDR risk score based on signature mutations.

        Returns:
            risk_score: 0-1 probability of MDR
            detected_markers: list of detected MDR markers
        """
        detected = []
        weighted_score = 0

        for mut in mutations:
            # Extract just the mutation code (e.g., L63P from various formats)
            mut_clean = mut.strip().upper()

            # Check each signature mutation
            for sig_mut, prevalence in self.signature_mutations.items():
                if sig_mut in mut_clean or mut_clean.endswith(sig_mut[-1]) and mut_clean[0] == sig_mut[0]:
                    detected.append(sig_mut)
                    weighted_score += prevalence
                    break

        # Normalize score
        max_possible = sum(self.signature_mutations.values())
        risk_score = min(1.0, weighted_score / (max_possible * 0.3))  # Scale to 0-1

        return risk_score, detected

    def screen_sequence(self, mutation_list: str) -> dict:
        """Screen a single sequence for MDR risk."""
        mutations = self.parse_mutation_list(mutation_list)
        risk_score, detected = self.compute_mdr_risk_score(mutations)

        # Risk classification
        if risk_score >= 0.7:
            risk_level = "HIGH"
            action = "Immediate resistance testing recommended. Consider alternative regimen."
        elif risk_score >= 0.4:
            risk_level = "MODERATE"
            action = "Enhanced monitoring recommended. Consider genotypic testing."
        elif risk_score >= 0.2:
            risk_level = "LOW"
            action = "Standard monitoring. Watch for emerging resistance."
        else:
            risk_level = "MINIMAL"
            action = "Continue current regimen with routine monitoring."

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "detected_markers": detected,
            "n_markers": len(detected),
            "recommended_action": action,
            "all_mutations": mutations,
        }

    def screen_batch(self, mutation_lists: list[str]) -> pd.DataFrame:
        """Screen multiple sequences."""
        results = []
        for i, mut_list in enumerate(mutation_lists):
            result = self.screen_sequence(mut_list)
            result["sequence_id"] = i
            results.append(result)
        return pd.DataFrame(results)

    def generate_report(self, hivdb_path: Path) -> dict:
        """Generate MDR screening report from HIVDB data."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "screening_summary": {},
            "high_risk_sequences": [],
            "mutation_frequency": {},
        }

        if not hivdb_path.exists():
            return report

        # Load data
        dfs = []
        for f in hivdb_path.glob("stanford_hivdb_*.txt"):
            try:
                df = pd.read_csv(f, sep="\t")
                dfs.append(df)
            except:
                pass

        if not dfs:
            return report

        combined = pd.concat(dfs, ignore_index=True)

        if "CompMutList" not in combined.columns:
            return report

        # Screen all sequences
        results = self.screen_batch(combined["CompMutList"].tolist())

        # Summary
        report["screening_summary"] = {
            "total_screened": len(results),
            "high_risk": int((results["risk_level"] == "HIGH").sum()),
            "moderate_risk": int((results["risk_level"] == "MODERATE").sum()),
            "low_risk": int((results["risk_level"] == "LOW").sum()),
            "minimal_risk": int((results["risk_level"] == "MINIMAL").sum()),
            "mean_risk_score": float(results["risk_score"].mean()),
        }

        # High risk sequences
        high_risk = results[results["risk_level"] == "HIGH"].head(20)
        for _, row in high_risk.iterrows():
            report["high_risk_sequences"].append(
                {
                    "sequence_id": int(row["sequence_id"]),
                    "risk_score": float(row["risk_score"]),
                    "n_markers": int(row["n_markers"]),
                    "markers": row["detected_markers"],
                }
            )

        # Mutation frequency in high-risk
        all_markers = []
        for markers in results[results["risk_level"] == "HIGH"]["detected_markers"]:
            all_markers.extend(markers)
        from collections import Counter

        report["mutation_frequency"] = dict(Counter(all_markers).most_common(10))

        return report


# =============================================================================
# TOOL 3: TAT-TARGETING DRUG ANALYZER
# =============================================================================


class TatDrugAnalyzer:
    """Analyze Tat-targeting drug opportunities."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.ppi_data = None
        self.tat_interactions = None

    def load_data(self):
        """Load PPI data."""
        ppi_path = (
            self.data_dir / "external" / "huggingface" / "human_hiv_ppi" / "data" / "train-00000-of-00001.parquet"
        )

        if not ppi_path.exists() or not HAS_PARQUET:
            return False

        self.ppi_data = pq.read_table(ppi_path).to_pandas()
        self.tat_interactions = self.ppi_data[self.ppi_data["hiv_protein_name"] == "Tat"]
        return True

    def classify_druggability(self, protein_name: str) -> dict:
        """Classify protein druggability."""
        if not isinstance(protein_name, str):
            return {"family": "unknown", "druggability": "unknown", "score": 0}

        name_lower = protein_name.lower()

        for family, info in DRUGGABLE_FAMILIES.items():
            if family in name_lower:
                score = {"high": 1.0, "medium": 0.6, "low": 0.3}.get(info["druggability"], 0)
                return {
                    "family": family,
                    "class": info["class"],
                    "druggability": info["druggability"],
                    "score": score,
                }

        return {"family": "other", "class": "unknown", "druggability": "unknown", "score": 0.2}

    def analyze_tat_targets(self) -> pd.DataFrame:
        """Analyze Tat interaction targets."""
        if self.tat_interactions is None and not self.load_data():
            return pd.DataFrame()

        results = []
        for _, row in self.tat_interactions.iterrows():
            human_protein = row["human_protein_name"]
            druggability = self.classify_druggability(human_protein)

            results.append(
                {
                    "human_protein": human_protein,
                    "interaction_type": row.get("interaction_type", "unknown"),
                    "description": row.get("description", "")[:200],
                    "family": druggability["family"],
                    "druggability": druggability["druggability"],
                    "druggability_score": druggability["score"],
                }
            )

        df = pd.DataFrame(results)

        # Aggregate by protein (may have multiple interaction types)
        if len(df) > 0:
            agg_df = (
                df.groupby("human_protein")
                .agg(
                    {
                        "interaction_type": lambda x: ", ".join(set(x)),
                        "family": "first",
                        "druggability": "first",
                        "druggability_score": "first",
                    }
                )
                .reset_index()
            )
            return agg_df.sort_values("druggability_score", ascending=False)

        return df

    def identify_drug_opportunities(self) -> dict:
        """Identify specific drug development opportunities."""
        targets = self.analyze_tat_targets()

        opportunities = {
            "kinase_inhibitors": [],
            "receptor_antagonists": [],
            "protease_inhibitors": [],
            "other_targets": [],
        }

        for _, row in targets.iterrows():
            target_info = {
                "protein": row["human_protein"],
                "interaction_type": row["interaction_type"],
                "druggability_score": row["druggability_score"],
            }

            family = row["family"]
            if family == "kinase":
                opportunities["kinase_inhibitors"].append(target_info)
            elif family == "receptor":
                opportunities["receptor_antagonists"].append(target_info)
            elif family == "protease":
                opportunities["protease_inhibitors"].append(target_info)
            elif row["druggability_score"] > 0.3:
                opportunities["other_targets"].append(target_info)

        # Limit each category
        for key in opportunities:
            opportunities[key] = opportunities[key][:20]

        return opportunities

    def generate_report(self) -> dict:
        """Generate Tat drug targeting report."""
        if self.tat_interactions is None:
            self.load_data()

        targets = self.analyze_tat_targets()
        opportunities = self.identify_drug_opportunities()

        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tat_interactions": len(self.tat_interactions) if self.tat_interactions is not None else 0,
                "unique_human_targets": len(targets),
                "high_druggability_targets": int((targets["druggability"] == "high").sum()) if len(targets) > 0 else 0,
            },
            "top_druggable_targets": [],
            "drug_opportunities": opportunities,
            "recommendations": [],
        }

        # Top targets
        for _, row in targets.head(15).iterrows():
            report["top_druggable_targets"].append(
                {
                    "protein": row["human_protein"],
                    "family": row["family"],
                    "druggability": row["druggability"],
                    "score": float(row["druggability_score"]),
                }
            )

        # Recommendations
        n_kinases = len(opportunities["kinase_inhibitors"])
        if n_kinases > 0:
            report["recommendations"].append(
                f"Consider repurposing existing kinase inhibitors - {n_kinases} kinase targets identified"
            )

        n_receptors = len(opportunities["receptor_antagonists"])
        if n_receptors > 0:
            report["recommendations"].append(
                f"Receptor-based therapies promising - {n_receptors} receptor targets identified"
            )

        report["recommendations"].append("Host-directed therapy via Tat targets avoids viral resistance evolution")

        return report


# =============================================================================
# TOOL 4: UNIFIED PREDICTION PIPELINE
# =============================================================================


class UnifiedPredictor:
    """Unified prediction pipeline with p-adic encoding."""

    def __init__(self):
        self.tropism_model = None
        self.resistance_model = None
        self.scaler = StandardScaler()

    def padic_encode_sequence(self, seq: str, max_len: int = 35) -> np.ndarray:
        """Encode sequence with p-adic inspired features."""
        encoding = np.zeros((max_len, 10))

        for i, aa in enumerate(seq[:max_len]):
            if aa == "-" or aa not in "ACDEFGHIKLMNPQRSTVWY":
                continue

            for g_idx, (_group_name, group_aas) in enumerate(AA_GROUPS.items()):
                if aa in group_aas:
                    encoding[i, 0] = g_idx / 6
                    encoding[i, 1] = group_aas.index(aa) / len(group_aas)
                    encoding[i, 2 + min(g_idx, 5)] = 1
                    break

            encoding[i, 8] = i / max_len
            encoding[i, 9] = 1 if i in [11, 24, 25] else 0

        return encoding.flatten()

    def onehot_encode_sequence(self, seq: str, max_len: int = 35) -> np.ndarray:
        """Standard one-hot encoding."""
        encoding = np.zeros((max_len, len(AA_ORDER)))
        for i, aa in enumerate(seq[:max_len]):
            if aa in AA_ORDER:
                encoding[i, AA_ORDER.index(aa)] = 1
        return encoding.flatten()

    def combined_encode(self, seq: str, max_len: int = 35) -> np.ndarray:
        """Combined p-adic + one-hot encoding."""
        padic = self.padic_encode_sequence(seq, max_len)
        onehot = self.onehot_encode_sequence(seq, max_len)
        return np.concatenate([padic, onehot])

    def train_tropism_model(self, data_dir: Path) -> bool:
        """Train tropism prediction model."""
        v3_path = data_dir / "external" / "huggingface" / "HIV_V3_coreceptor" / "data" / "train-00000-of-00001.parquet"

        if not v3_path.exists() or not HAS_PARQUET:
            return False

        df = pq.read_table(v3_path).to_pandas()

        X = np.array([self.combined_encode(s) for s in df["sequence"]])
        y = df["CXCR4"].astype(int).values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.tropism_model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.tropism_model.fit(X_train_scaled, y_train)

        y_pred = self.tropism_model.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, y_pred)

        print(f"  Tropism model trained: AUC = {auc:.4f}")
        return True

    def predict_tropism(self, sequence: str) -> dict:
        """Predict tropism for a sequence."""
        if self.tropism_model is None:
            return {"error": "Model not trained"}

        X = self.combined_encode(sequence).reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        proba = self.tropism_model.predict_proba(X_scaled)[0]

        return {
            "sequence": sequence[:30] + "..." if len(sequence) > 30 else sequence,
            "R5_probability": float(proba[0]),
            "X4_probability": float(proba[1]),
            "predicted_tropism": "X4" if proba[1] > 0.5 else "R5",
            "confidence": float(max(proba)),
        }

    def batch_predict(self, sequences: list[str]) -> pd.DataFrame:
        """Batch predict tropism."""
        results = [self.predict_tropism(seq) for seq in sequences]
        return pd.DataFrame(results)


# =============================================================================
# TOOL 5: CLINICAL DECISION SUPPORT
# =============================================================================


class ClinicalDecisionSupport:
    """Generate clinical decision support reports."""

    def __init__(self, results_dir: Path, data_dir: Path):
        self.results_dir = results_dir
        self.data_dir = data_dir

        self.vaccine_prioritizer = VaccinePrioritizer(results_dir)
        self.mdr_screener = MDRScreener()
        self.tat_analyzer = TatDrugAnalyzer(data_dir)
        self.predictor = UnifiedPredictor()

    def generate_comprehensive_report(self) -> dict:
        """Generate comprehensive clinical decision support report."""
        print("\n  Generating vaccine prioritization...")
        vaccine_report = self.vaccine_prioritizer.generate_report()

        print("  Generating MDR screening report...")
        hivdb_path = self.data_dir.parent / "research" / "datasets"
        mdr_report = self.mdr_screener.generate_report(hivdb_path)

        print("  Generating Tat drug analysis...")
        tat_report = self.tat_analyzer.generate_report()

        print("  Training prediction models...")
        self.predictor.train_tropism_model(self.data_dir)

        # Compile comprehensive report
        report = {
            "title": "HIV Clinical Decision Support Report",
            "generated": datetime.now().isoformat(),
            "sections": {
                "vaccine_development": vaccine_report,
                "resistance_monitoring": mdr_report,
                "host_directed_therapy": tat_report,
            },
            "executive_summary": self._generate_executive_summary(vaccine_report, mdr_report, tat_report),
            "action_items": self._generate_action_items(vaccine_report, mdr_report, tat_report),
        }

        return report

    def _generate_executive_summary(self, vaccine, mdr, tat) -> list[str]:
        """Generate executive summary."""
        summary = []

        # Vaccine
        if vaccine.get("top_candidates"):
            top = vaccine["top_candidates"][0]
            summary.append(
                f"TOP VACCINE CANDIDATE: {top['epitope']} from {top['protein']} "
                f"(priority score: {top['priority_score']:.3f})"
            )

        # MDR
        if mdr.get("screening_summary"):
            ss = mdr["screening_summary"]
            high_pct = ss.get("high_risk", 0) / max(ss.get("total_screened", 1), 1) * 100
            summary.append(
                f"MDR ALERT: {ss.get('high_risk', 0)} high-risk sequences detected "
                f"({high_pct:.1f}% of screened population)"
            )

        # Tat
        if tat.get("summary"):
            n_targets = tat["summary"].get("high_druggability_targets", 0)
            summary.append(
                f"DRUG TARGETS: {n_targets} high-druggability Tat targets identified "
                f"for host-directed therapy development"
            )

        return summary

    def _generate_action_items(self, vaccine, mdr, tat) -> list[dict]:
        """Generate prioritized action items."""
        actions = []

        # High priority: MDR screening
        if mdr.get("screening_summary", {}).get("high_risk", 0) > 0:
            actions.append(
                {
                    "priority": "HIGH",
                    "category": "Resistance Monitoring",
                    "action": "Implement L63P/L90M screening in clinical genotyping",
                    "rationale": "79.5% of MDR cases carry L63P mutation",
                }
            )

        # Medium priority: Vaccine development
        if vaccine.get("top_candidates"):
            actions.append(
                {
                    "priority": "MEDIUM",
                    "category": "Vaccine Development",
                    "action": f"Advance {vaccine['top_candidates'][0]['epitope']} to preclinical",
                    "rationale": "Highest stability score + population coverage",
                }
            )

        # Research priority: Host-directed therapy
        if tat.get("drug_opportunities", {}).get("kinase_inhibitors"):
            n_kinases = len(tat["drug_opportunities"]["kinase_inhibitors"])
            actions.append(
                {
                    "priority": "RESEARCH",
                    "category": "Drug Development",
                    "action": f"Screen {n_kinases} kinase inhibitors for Tat disruption",
                    "rationale": "Host-directed approach avoids resistance",
                }
            )

        return actions


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 70)
    print("HIV CLINICAL APPLICATIONS PIPELINE")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    data_dir = PROJECT_ROOT / "data"
    results_dir = PROJECT_ROOT / "research" / "bioinformatics" / "codon_encoder_research" / "hiv" / "results"
    output_dir = PROJECT_ROOT / "results" / "clinical_applications"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize clinical decision support
    cds = ClinicalDecisionSupport(results_dir, data_dir)

    print("\n" + "=" * 70)
    print("GENERATING CLINICAL DECISION SUPPORT REPORT")
    print("=" * 70)

    report = cds.generate_comprehensive_report()

    # Save JSON report
    json_path = output_dir / "clinical_decision_support.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nJSON report saved: {json_path}")

    # Generate markdown report
    md_path = output_dir / "CLINICAL_REPORT.md"
    with open(md_path, "w") as f:
        f.write("# HIV Clinical Decision Support Report\n\n")
        f.write(f"Generated: {report['generated']}\n\n")

        f.write("## Executive Summary\n\n")
        for item in report.get("executive_summary", []):
            f.write(f"- **{item}**\n")
        f.write("\n")

        f.write("## Priority Action Items\n\n")
        for action in report.get("action_items", []):
            f.write(f"### [{action['priority']}] {action['category']}\n")
            f.write(f"**Action**: {action['action']}\n\n")
            f.write(f"**Rationale**: {action['rationale']}\n\n")

        f.write("## Vaccine Development\n\n")
        vaccine = report["sections"]["vaccine_development"]
        if vaccine.get("top_candidates"):
            f.write("### Top Candidates\n\n")
            f.write("| Epitope | Protein | Priority Score | Stability | Coverage |\n")
            f.write("|---------|---------|----------------|-----------|----------|\n")
            for c in vaccine["top_candidates"][:10]:
                f.write(f"| {c['epitope']} | {c['protein']} | {c['priority_score']:.3f} | ")
                f.write(f"{c['stability_score']:.1f} | {c['population_coverage']:.2f} |\n")
        f.write("\n")

        f.write("## Resistance Monitoring\n\n")
        mdr = report["sections"]["resistance_monitoring"]
        if mdr.get("screening_summary"):
            ss = mdr["screening_summary"]
            f.write(f"- Total screened: {ss.get('total_screened', 0)}\n")
            f.write(f"- High risk: {ss.get('high_risk', 0)}\n")
            f.write(f"- Mean risk score: {ss.get('mean_risk_score', 0):.3f}\n\n")

        if mdr.get("mutation_frequency"):
            f.write("### MDR Signature Mutations\n\n")
            f.write("| Mutation | Frequency |\n")
            f.write("|----------|----------|\n")
            for mut, freq in list(mdr["mutation_frequency"].items())[:10]:
                f.write(f"| {mut} | {freq} |\n")
        f.write("\n")

        f.write("## Host-Directed Therapy Targets\n\n")
        tat = report["sections"]["host_directed_therapy"]
        if tat.get("top_druggable_targets"):
            f.write("### Tat-Interacting Druggable Proteins\n\n")
            f.write("| Protein | Family | Druggability |\n")
            f.write("|---------|--------|-------------|\n")
            for t in tat["top_druggable_targets"][:15]:
                f.write(f"| {t['protein'][:50]} | {t['family']} | {t['druggability']} |\n")
        f.write("\n")

        if tat.get("recommendations"):
            f.write("### Recommendations\n\n")
            for rec in tat["recommendations"]:
                f.write(f"- {rec}\n")

    print(f"Markdown report saved: {md_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("EXECUTIVE SUMMARY")
    print("=" * 70)
    for item in report.get("executive_summary", []):
        print(f"\n  * {item}")

    print("\n" + "-" * 70)
    print("PRIORITY ACTION ITEMS")
    print("-" * 70)
    for action in report.get("action_items", []):
        print(f"\n  [{action['priority']}] {action['category']}")
        print(f"    Action: {action['action']}")
        print(f"    Rationale: {action['rationale']}")

    print("\n" + "=" * 70)
    print("CLINICAL APPLICATIONS PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
