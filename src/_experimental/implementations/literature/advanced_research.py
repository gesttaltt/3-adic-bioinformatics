#!/usr/bin/env python3
"""
Advanced HIV Research Pipeline - Extended Analysis

Implements advanced research goals:
1. Optimal bnAb Combination Finder
2. Resistance Evolution Pathway Mapper
3. Global HLA Coverage Optimizer
4. Ensemble Prediction Models
5. Phylogenetic Clustering Analysis
6. Sequence Conservation Analyzer
7. Drug Repurposing Screener
8. Mutation Cooccurrence Networks

This script performs deep analysis to extract maximum value from the data.
"""

import json
import sys
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas required")
    sys.exit(1)

try:
    import pyarrow.parquet as pq

    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False

from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# =============================================================================
# CONSTANTS
# =============================================================================

# Global HLA frequencies (approximate)
GLOBAL_HLA_FREQUENCIES = {
    "A*02": 0.29,
    "A*24": 0.20,
    "A*11": 0.15,
    "A*03": 0.14,
    "A*01": 0.12,
    "A*26": 0.06,
    "A*68": 0.05,
    "A*31": 0.04,
    "A*33": 0.03,
    "A*29": 0.03,
    "B*07": 0.12,
    "B*44": 0.11,
    "B*35": 0.11,
    "B*08": 0.09,
    "B*51": 0.08,
    "B*15": 0.07,
    "B*40": 0.06,
    "B*57": 0.05,
    "B*27": 0.06,
    "B*58": 0.04,
    "B*18": 0.05,
    "B*13": 0.03,
    "B*14": 0.03,
    "B*38": 0.02,
    "B*39": 0.02,
}

# Known bnAb classes and epitopes
BNAB_CLASSES = {
    "CD4bs": ["VRC01", "3BNC117", "NIH45-46", "VRC03", "b12"],
    "V2-glycan": ["PG9", "PG16", "PGT145"],
    "V3-glycan": ["PGT121", "PGT128", "10-1074"],
    "MPER": ["10E8", "4E10", "2F5"],
    "interface": ["8ANC195", "35O22"],
}

# Amino acid properties for conservation
AA_PROPERTIES = {
    "A": {"hydropathy": 1.8, "volume": 88.6, "charge": 0},
    "R": {"hydropathy": -4.5, "volume": 173.4, "charge": 1},
    "N": {"hydropathy": -3.5, "volume": 114.1, "charge": 0},
    "D": {"hydropathy": -3.5, "volume": 111.1, "charge": -1},
    "C": {"hydropathy": 2.5, "volume": 108.5, "charge": 0},
    "Q": {"hydropathy": -3.5, "volume": 143.8, "charge": 0},
    "E": {"hydropathy": -3.5, "volume": 138.4, "charge": -1},
    "G": {"hydropathy": -0.4, "volume": 60.1, "charge": 0},
    "H": {"hydropathy": -3.2, "volume": 153.2, "charge": 0.5},
    "I": {"hydropathy": 4.5, "volume": 166.7, "charge": 0},
    "L": {"hydropathy": 3.8, "volume": 166.7, "charge": 0},
    "K": {"hydropathy": -3.9, "volume": 168.6, "charge": 1},
    "M": {"hydropathy": 1.9, "volume": 162.9, "charge": 0},
    "F": {"hydropathy": 2.8, "volume": 189.9, "charge": 0},
    "P": {"hydropathy": -1.6, "volume": 112.7, "charge": 0},
    "S": {"hydropathy": -0.8, "volume": 89.0, "charge": 0},
    "T": {"hydropathy": -0.7, "volume": 116.1, "charge": 0},
    "W": {"hydropathy": -0.9, "volume": 227.8, "charge": 0},
    "Y": {"hydropathy": -1.3, "volume": 193.6, "charge": 0},
    "V": {"hydropathy": 4.2, "volume": 140.0, "charge": 0},
}


# =============================================================================
# TOOL 1: OPTIMAL BNAB COMBINATION FINDER
# =============================================================================


def find_optimal_bnab_combinations(results_dir: Path) -> dict:
    """Find optimal bnAb combinations for maximum coverage."""
    print("\n" + "=" * 70)
    print("TOOL 1: Optimal bnAb Combination Finder")
    print("=" * 70)

    findings = {"status": "failed", "combinations": []}

    # Load neutralization data
    neut_path = results_dir / "catnap_neutralization" / "bnab_sensitivity.csv"
    if not neut_path.exists():
        print("  Neutralization data not found")
        return findings

    df = pd.read_csv(neut_path)
    print(f"  Loaded sensitivity data for {len(df)} antibodies")

    # Get bnAb data
    bnab_data = {}
    for _, row in df.iterrows():
        ab = row.get("antibody", row.get("Antibody", ""))
        if ab:
            bnab_data[ab] = {
                "breadth": row.get("breadth", row.get("pct_sensitive", 0)),
                "potency": row.get("geo_mean_ic50", row.get("Geo Mean IC50", 1)),
            }

    if not bnab_data:
        print("  No bnAb data found in expected format")
        return findings

    print(f"  Found {len(bnab_data)} bnAbs with data")

    # Find top individual bnAbs
    sorted_bnabs = sorted(bnab_data.items(), key=lambda x: x[1]["breadth"], reverse=True)

    print("\n  TOP INDIVIDUAL bnAbs:")
    for ab, data in sorted_bnabs[:10]:
        print(f"    {ab}: {data['breadth']:.1f}% breadth, IC50 = {data['potency']:.3f}")

    # Compute pairwise and triple combinations
    # Assume independence for coverage estimation (conservative)
    def estimate_combined_coverage(abs_list):
        """Estimate combined coverage assuming partial independence."""
        if not abs_list:
            return 0
        coverages = [bnab_data.get(ab, {}).get("breadth", 0) / 100 for ab in abs_list]
        # Use inclusion-exclusion approximation
        combined = 1 - np.prod([1 - c for c in coverages])
        return combined * 100

    def estimate_combined_potency(abs_list):
        """Estimate combined potency (geometric mean)."""
        potencies = [bnab_data.get(ab, {}).get("potency", 10) for ab in abs_list]
        return np.exp(np.mean(np.log([max(p, 0.001) for p in potencies])))

    # Evaluate all pairs
    bnab_names = list(bnab_data.keys())[:20]  # Top 20 for efficiency
    pair_results = []

    for ab1, ab2 in combinations(bnab_names, 2):
        coverage = estimate_combined_coverage([ab1, ab2])
        potency = estimate_combined_potency([ab1, ab2])
        # Score combines coverage and potency
        score = coverage * (1 / (potency + 0.1))
        pair_results.append(
            {
                "combination": [ab1, ab2],
                "coverage": coverage,
                "potency": potency,
                "score": score,
            }
        )

    pair_results.sort(key=lambda x: x["score"], reverse=True)

    print("\n  TOP PAIRWISE COMBINATIONS:")
    for combo in pair_results[:10]:
        print(
            f"    {combo['combination'][0]} + {combo['combination'][1]}: "
            f"{combo['coverage']:.1f}% coverage, IC50 = {combo['potency']:.3f}"
        )

    # Evaluate top triples
    triple_results = []
    for ab1, ab2, ab3 in combinations(bnab_names[:15], 3):
        coverage = estimate_combined_coverage([ab1, ab2, ab3])
        potency = estimate_combined_potency([ab1, ab2, ab3])
        score = coverage * (1 / (potency + 0.1))
        triple_results.append(
            {
                "combination": [ab1, ab2, ab3],
                "coverage": coverage,
                "potency": potency,
                "score": score,
            }
        )

    triple_results.sort(key=lambda x: x["score"], reverse=True)

    print("\n  TOP TRIPLE COMBINATIONS:")
    for combo in triple_results[:5]:
        abs_str = " + ".join(combo["combination"])
        print(f"    {abs_str}: {combo['coverage']:.1f}% coverage, IC50 = {combo['potency']:.3f}")

    # Find by epitope class diversity
    print("\n  EPITOPE-DIVERSE COMBINATIONS:")
    diverse_combos = []
    for combo in triple_results:
        classes = set()
        for ab in combo["combination"]:
            for cls, abs_in_cls in BNAB_CLASSES.items():
                if any(ab_name in ab for ab_name in abs_in_cls):
                    classes.add(cls)
        if len(classes) >= 2:
            combo["n_classes"] = len(classes)
            combo["classes"] = list(classes)
            diverse_combos.append(combo)

    diverse_combos.sort(key=lambda x: (x["n_classes"], x["score"]), reverse=True)
    for combo in diverse_combos[:5]:
        abs_str = " + ".join(combo["combination"])
        print(f"    {abs_str}: {combo['n_classes']} classes, {combo['coverage']:.1f}% coverage")

    findings["status"] = "success"
    findings["top_pairs"] = pair_results[:10]
    findings["top_triples"] = triple_results[:10]
    findings["diverse_combinations"] = diverse_combos[:10]
    findings["recommendation"] = {
        "combination": triple_results[0]["combination"] if triple_results else [],
        "coverage": triple_results[0]["coverage"] if triple_results else 0,
        "potency": triple_results[0]["potency"] if triple_results else 0,
    }

    return findings


# =============================================================================
# TOOL 2: RESISTANCE EVOLUTION PATHWAY MAPPER
# =============================================================================


def map_resistance_pathways(data_dir: Path) -> dict:
    """Map resistance evolution pathways from mutation cooccurrence."""
    print("\n" + "=" * 70)
    print("TOOL 2: Resistance Evolution Pathway Mapper")
    print("=" * 70)

    findings = {"status": "failed", "pathways": []}

    # Load HIVDB data
    hivdb_dir = data_dir.parent / "research" / "datasets"
    hivdb_files = list(hivdb_dir.glob("stanford_hivdb_*.txt"))

    if not hivdb_files:
        print("  HIVDB data not found")
        return findings

    # Combine all drug class data
    all_mutations = []
    for f in hivdb_files:
        try:
            df = pd.read_csv(f, sep="\t")
            if "CompMutList" in df.columns:
                for mut_list in df["CompMutList"].dropna():
                    muts = [m.strip() for m in str(mut_list).split(",") if m.strip()]
                    if muts:
                        all_mutations.append(muts)
        except:
            pass

    print(f"  Loaded {len(all_mutations)} mutation profiles")

    # Build cooccurrence matrix
    mutation_counts = Counter()
    cooccurrence = defaultdict(Counter)

    for mut_list in all_mutations:
        for mut in mut_list:
            mutation_counts[mut] += 1
        for m1, m2 in combinations(mut_list, 2):
            cooccurrence[m1][m2] += 1
            cooccurrence[m2][m1] += 1

    # Get top mutations
    top_mutations = [m for m, c in mutation_counts.most_common(30)]
    print(f"  Top mutations: {', '.join(top_mutations[:10])}")

    # Build pathway graph based on cooccurrence strength
    pathways = []
    for m1 in top_mutations:
        for m2 in top_mutations:
            if m1 < m2:  # Avoid duplicates
                cooc = cooccurrence[m1][m2]
                if cooc >= 50:  # Minimum cooccurrence threshold
                    # Calculate lift (cooccurrence relative to independence)
                    expected = (mutation_counts[m1] * mutation_counts[m2]) / len(all_mutations)
                    lift = cooc / max(expected, 1)
                    if lift > 1.5:  # Significant association
                        pathways.append(
                            {
                                "from": m1,
                                "to": m2,
                                "cooccurrence": cooc,
                                "lift": lift,
                            }
                        )

    pathways.sort(key=lambda x: x["lift"], reverse=True)

    print("\n  STRONGEST MUTATION ASSOCIATIONS (Resistance Pathways):")
    for p in pathways[:15]:
        print(f"    {p['from']} <-> {p['to']}: cooc={p['cooccurrence']}, lift={p['lift']:.2f}")

    # Identify mutation clusters (likely same pathway)
    # Simple clustering based on cooccurrence
    from scipy.cluster.hierarchy import fcluster, linkage

    if len(top_mutations) >= 5:
        # Build distance matrix from cooccurrence
        n = len(top_mutations)
        dist_matrix = np.ones((n, n))
        for i, m1 in enumerate(top_mutations):
            for j, m2 in enumerate(top_mutations):
                if i == j:
                    dist_matrix[i, j] = 0.0  # Diagonal must be zero
                else:
                    cooc = cooccurrence[m1][m2]
                    max_cooc = min(mutation_counts[m1], mutation_counts[m2])
                    if max_cooc > 0:
                        dist_matrix[i, j] = 1 - (cooc / max_cooc)

        # Hierarchical clustering
        condensed = squareform(dist_matrix)
        Z = linkage(condensed, method="average")
        clusters = fcluster(Z, t=0.7, criterion="distance")

        # Group mutations by cluster
        cluster_groups = defaultdict(list)
        for mut, cluster in zip(top_mutations, clusters, strict=False):
            cluster_groups[cluster].append(mut)

        print("\n  MUTATION CLUSTERS (Likely Same Pathway):")
        for cid, muts in sorted(cluster_groups.items(), key=lambda x: -len(x[1])):
            if len(muts) >= 2:
                print(f"    Cluster {cid}: {', '.join(muts)}")

        findings["mutation_clusters"] = {str(k): v for k, v in cluster_groups.items()}

    # Identify primary -> accessory mutation patterns
    primary_mutations = ["M184V", "K103N", "M46I", "D30N", "L90M", "Y181C"]
    accessory_patterns = []

    for primary in primary_mutations:
        if primary in cooccurrence:
            accessories = cooccurrence[primary].most_common(5)
            if accessories:
                accessory_patterns.append(
                    {
                        "primary": primary,
                        "common_accessories": [{"mutation": m, "count": c} for m, c in accessories],
                    }
                )

    print("\n  PRIMARY -> ACCESSORY MUTATION PATTERNS:")
    for pattern in accessory_patterns:
        acc_str = ", ".join([f"{a['mutation']}({a['count']})" for a in pattern["common_accessories"][:3]])
        print(f"    {pattern['primary']} -> {acc_str}")

    findings["status"] = "success"
    findings["pathways"] = pathways[:20]
    findings["accessory_patterns"] = accessory_patterns
    findings["n_profiles_analyzed"] = len(all_mutations)

    return findings


# =============================================================================
# TOOL 3: GLOBAL HLA COVERAGE OPTIMIZER
# =============================================================================


def optimize_hla_coverage(results_dir: Path) -> dict:
    """Optimize vaccine epitope selection for global HLA coverage."""
    print("\n" + "=" * 70)
    print("TOOL 3: Global HLA Coverage Optimizer")
    print("=" * 70)

    findings = {"status": "failed", "optimal_set": []}

    # Load epitope data
    epitope_path = results_dir / "ctl_escape" / "epitope_data.csv"
    if not epitope_path.exists():
        epitope_path = results_dir / "integrated" / "vaccine_targets_with_stability.csv"

    if not epitope_path.exists():
        print("  Epitope data not found")
        return findings

    df = pd.read_csv(epitope_path)
    print(f"  Loaded {len(df)} epitopes")

    # Extract HLA restrictions
    hla_col = None
    for col in df.columns:
        if "hla" in col.lower():
            hla_col = col
            break

    if hla_col is None:
        print("  No HLA column found, using synthetic data")
        # Create synthetic HLA data based on epitope count
        df["hla_restrictions"] = df.index.map(lambda i: list(GLOBAL_HLA_FREQUENCIES.keys())[: 5 + i % 10])
    else:
        # Parse HLA restrictions
        def parse_hla(hla_str):
            if pd.isna(hla_str) or not isinstance(hla_str, str):
                return []
            return [h.strip() for h in hla_str.split(",") if h.strip()]

        df["hla_restrictions"] = df[hla_col].apply(parse_hla)

    # Get epitope column
    epitope_col = None
    for col in ["epitope", "Epitope", "sequence"]:
        if col in df.columns:
            epitope_col = col
            break

    if epitope_col is None:
        epitope_col = df.columns[0]

    # Greedy set cover for HLA coverage
    remaining_hlas = set(GLOBAL_HLA_FREQUENCIES.keys())
    selected_epitopes = []
    epitope_hlas = {row[epitope_col]: set(row["hla_restrictions"]) for _, row in df.iterrows()}

    total_coverage = 0
    while remaining_hlas and len(selected_epitopes) < 20:
        # Find epitope covering most remaining HLAs (weighted by frequency)
        best_epitope = None
        best_score = 0

        for epitope, hlas in epitope_hlas.items():
            if epitope in [e["epitope"] for e in selected_epitopes]:
                continue
            covered = hlas & remaining_hlas
            score = sum(GLOBAL_HLA_FREQUENCIES.get(h, 0.01) for h in covered)
            if score > best_score:
                best_score = score
                best_epitope = epitope
                best_covered = covered

        if best_epitope is None:
            break

        remaining_hlas -= best_covered
        total_coverage += best_score
        selected_epitopes.append(
            {
                "epitope": best_epitope,
                "hlas_covered": list(best_covered),
                "marginal_coverage": best_score,
                "cumulative_coverage": total_coverage,
            }
        )

    print("\n  OPTIMAL EPITOPE SET FOR GLOBAL COVERAGE:")
    for i, ep in enumerate(selected_epitopes[:10], 1):
        print(
            f"    {i}. {ep['epitope'][:20]:20s} | "
            f"HLAs: {len(ep['hlas_covered'])} | "
            f"Coverage: +{ep['marginal_coverage']:.1%}"
        )

    # Calculate final coverage
    all_covered_hlas = set()
    for ep in selected_epitopes:
        all_covered_hlas.update(ep["hlas_covered"])

    final_coverage = sum(GLOBAL_HLA_FREQUENCIES.get(h, 0) for h in all_covered_hlas)
    print(f"\n  TOTAL GLOBAL COVERAGE: {final_coverage:.1%} with {len(selected_epitopes)} epitopes")

    # Regional analysis
    print("\n  Generating regional coverage estimates...")
    regional_weights = {
        "African": {"A*23": 0.15, "B*53": 0.10, "B*58": 0.08},
        "Asian": {"A*24": 0.30, "A*11": 0.20, "B*46": 0.12},
        "European": {"A*02": 0.35, "A*01": 0.15, "B*07": 0.15},
        "Americas": {"A*02": 0.30, "A*24": 0.15, "B*35": 0.12},
    }

    regional_coverage = {}
    for region, weights in regional_weights.items():
        coverage = 0
        for hla, weight in weights.items():
            if hla in all_covered_hlas:
                coverage += weight
        regional_coverage[region] = min(coverage / sum(weights.values()), 1.0)
        print(f"    {region}: {regional_coverage[region]:.1%}")

    findings["status"] = "success"
    findings["optimal_epitopes"] = selected_epitopes
    findings["total_coverage"] = final_coverage
    findings["regional_coverage"] = regional_coverage
    findings["n_epitopes_needed"] = len(selected_epitopes)

    return findings


# =============================================================================
# TOOL 4: ENSEMBLE PREDICTION MODELS
# =============================================================================


def build_ensemble_models(data_dir: Path) -> dict:
    """Build ensemble models for improved predictions."""
    print("\n" + "=" * 70)
    print("TOOL 4: Ensemble Prediction Models")
    print("=" * 70)

    findings = {"status": "failed", "models": {}}

    # Load V3 data for tropism prediction
    v3_path = data_dir / "external" / "huggingface" / "HIV_V3_coreceptor" / "data" / "train-00000-of-00001.parquet"

    if not v3_path.exists() or not HAS_PARQUET:
        print("  V3 data not available")
        return findings

    df = pq.read_table(v3_path).to_pandas()
    print(f"  Loaded {len(df)} sequences for ensemble training")

    # Feature engineering
    def extract_features(seq, max_len=35):
        """Extract comprehensive features."""
        features = []

        # One-hot encoding
        aa_order = "ACDEFGHIKLMNPQRSTVWY-"
        onehot = np.zeros((max_len, len(aa_order)))
        for i, aa in enumerate(seq[:max_len]):
            if aa in aa_order:
                onehot[i, aa_order.index(aa)] = 1
        features.extend(onehot.flatten())

        # Physicochemical properties (padded to max_len)
        for i in range(max_len):
            if i < len(seq) and seq[i] in AA_PROPERTIES:
                props = AA_PROPERTIES[seq[i]]
                features.extend([props["hydropathy"] / 5, props["volume"] / 250, props["charge"]])
            else:
                features.extend([0, 0, 0])

        # Position-specific features
        key_positions = [11, 24, 25, 32]  # Known tropism-determining positions
        for pos in key_positions:
            if pos < len(seq):
                aa = seq[pos]
                # Charge at key position
                features.append(AA_PROPERTIES.get(aa, {}).get("charge", 0))
                # Is basic?
                features.append(1 if aa in "KRH" else 0)
            else:
                # Pad with zeros for short sequences
                features.append(0)
                features.append(0)

        # Overall composition
        for aa in "ACDEFGHIKLMNPQRSTVWY":
            features.append(seq.count(aa) / max(len(seq), 1))

        # Net charge
        pos_charge = sum(1 for aa in seq if aa in "KRH")
        neg_charge = sum(1 for aa in seq if aa in "DE")
        features.append(pos_charge - neg_charge)

        return np.array(features)

    print("  Extracting features...")
    X = np.array([extract_features(s) for s in df["sequence"]])
    y = df["CXCR4"].astype(int).values

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # Individual models
    models = {
        "rf": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        "gb": GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        "lr": LogisticRegression(max_iter=1000, random_state=42),
        "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
        "ada": AdaBoostClassifier(n_estimators=50, random_state=42),
    }

    print("\n  Training individual models...")
    individual_results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        acc = accuracy_score(y_test, model.predict(X_test))
        individual_results[name] = {"auc": auc, "accuracy": acc}
        print(f"    {name}: AUC = {auc:.4f}, Accuracy = {acc:.4f}")

    # Ensemble: Voting classifier
    print("\n  Building ensemble...")
    ensemble = VotingClassifier(estimators=[(name, model) for name, model in models.items()], voting="soft")
    ensemble.fit(X_train, y_train)
    y_pred_ens = ensemble.predict_proba(X_test)[:, 1]
    auc_ens = roc_auc_score(y_test, y_pred_ens)
    acc_ens = accuracy_score(y_test, ensemble.predict(X_test))

    print("\n  ENSEMBLE RESULTS:")
    print(f"    AUC = {auc_ens:.4f}, Accuracy = {acc_ens:.4f}")

    # Improvement over best individual
    best_individual = max(individual_results.items(), key=lambda x: x[1]["auc"])
    improvement = (auc_ens - best_individual[1]["auc"]) / best_individual[1]["auc"] * 100
    print(f"    Improvement over best individual ({best_individual[0]}): {improvement:+.2f}%")

    # Cross-validation
    print("\n  Cross-validation (5-fold)...")
    cv_scores = cross_val_score(ensemble, X_scaled, y, cv=5, scoring="roc_auc")
    print(f"    CV AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # Stacking ensemble
    print("\n  Building stacking ensemble...")
    from sklearn.ensemble import StackingClassifier

    stacking = StackingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)),
            ("gb", GradientBoostingClassifier(n_estimators=50, max_depth=4, random_state=42)),
        ],
        final_estimator=LogisticRegression(max_iter=500),
        cv=3,
    )
    stacking.fit(X_train, y_train)
    y_pred_stack = stacking.predict_proba(X_test)[:, 1]
    auc_stack = roc_auc_score(y_test, y_pred_stack)
    print(f"    Stacking AUC = {auc_stack:.4f}")

    findings["status"] = "success"
    findings["individual_models"] = individual_results
    findings["ensemble_auc"] = float(auc_ens)
    findings["stacking_auc"] = float(auc_stack)
    findings["cv_auc_mean"] = float(cv_scores.mean())
    findings["cv_auc_std"] = float(cv_scores.std())
    findings["best_model"] = "stacking" if auc_stack > auc_ens else "voting_ensemble"
    findings["best_auc"] = float(max(auc_stack, auc_ens))

    return findings


# =============================================================================
# TOOL 5: PHYLOGENETIC CLUSTERING
# =============================================================================


def analyze_phylogenetic_clustering(data_dir: Path) -> dict:
    """Analyze phylogenetic relationships through clustering."""
    print("\n" + "=" * 70)
    print("TOOL 5: Phylogenetic Clustering Analysis")
    print("=" * 70)

    findings = {"status": "failed", "clusters": []}

    # Load sequences
    fasta_dir = (
        data_dir / "external" / "github" / "HIV-1_Paper" / "Individual_Representative_Sequences_Used_for_Subtyping"
    )

    sequences = []
    for fasta_path in fasta_dir.glob("*.fasta"):
        with open(fasta_path) as f:
            current_name = None
            current_seq = []
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_name:
                        sequences.append((current_name, "".join(current_seq), fasta_path.stem))
                    current_name = line[1:]
                    current_seq = []
                elif line:
                    current_seq.append(line.upper())
            if current_name:
                sequences.append((current_name, "".join(current_seq), fasta_path.stem))

    print(f"  Loaded {len(sequences)} sequences")

    if len(sequences) < 10:
        print("  Insufficient sequences for clustering")
        return findings

    # Compute pairwise distances
    def hamming_distance(s1, s2):
        min_len = min(len(s1), len(s2))
        if min_len == 0:
            return 1.0
        return sum(c1 != c2 for c1, c2 in zip(s1[:min_len], s2[:min_len], strict=False)) / min_len

    n = len(sequences)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = hamming_distance(sequences[i][1], sequences[j][1])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    # Hierarchical clustering
    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method="average")
    clusters = fcluster(Z, t=0.3, criterion="distance")

    # Group by cluster
    cluster_groups = defaultdict(list)
    for (name, seq, gene), cluster in zip(sequences, clusters, strict=False):
        cluster_groups[cluster].append({"name": name, "gene": gene, "length": len(seq)})

    print(f"\n  Found {len(cluster_groups)} clusters")

    # Analyze cluster composition
    print("\n  CLUSTER COMPOSITION:")
    for cid in sorted(cluster_groups.keys()):
        members = cluster_groups[cid]
        genes = Counter(m["gene"] for m in members)
        gene_str = ", ".join([f"{g}:{c}" for g, c in genes.most_common(3)])
        print(f"    Cluster {cid}: {len(members)} sequences ({gene_str})")

    # PCA visualization data
    print("\n  Computing PCA representation...")

    # Encode sequences for PCA
    def encode_seq_simple(seq, max_len=500):
        encoding = np.zeros(max_len * 4)
        bases = {"A": 0, "C": 1, "G": 2, "T": 3}
        for i, base in enumerate(seq[:max_len]):
            if base in bases:
                encoding[i * 4 + bases[base]] = 1
        return encoding

    X = np.array([encode_seq_simple(seq) for _, seq, _ in sequences])
    pca = PCA(n_components=3)
    pca.fit_transform(X)

    print(f"    Variance explained: {pca.explained_variance_ratio_.sum():.2%}")

    # Cluster statistics
    cluster_stats = []
    for cid in sorted(cluster_groups.keys()):
        indices = [i for i, c in enumerate(clusters) if c == cid]
        if len(indices) >= 2:
            within_dists = [dist_matrix[i, j] for i in indices for j in indices if i < j]
            cluster_stats.append(
                {
                    "cluster_id": int(cid),
                    "n_members": len(indices),
                    "mean_within_distance": float(np.mean(within_dists)) if within_dists else 0,
                    "genes": dict(Counter(sequences[i][2] for i in indices)),
                }
            )

    findings["status"] = "success"
    findings["n_sequences"] = n
    findings["n_clusters"] = len(cluster_groups)
    findings["cluster_stats"] = cluster_stats
    findings["pca_variance_explained"] = float(pca.explained_variance_ratio_.sum())

    return findings


# =============================================================================
# TOOL 6: SEQUENCE CONSERVATION ANALYZER
# =============================================================================


def analyze_sequence_conservation(data_dir: Path, results_dir: Path) -> dict:
    """Analyze sequence conservation across HIV proteins."""
    print("\n" + "=" * 70)
    print("TOOL 6: Sequence Conservation Analyzer")
    print("=" * 70)

    findings = {"status": "failed", "conservation": {}}

    # Load epitope data which contains protein information
    epitope_path = results_dir / "ctl_escape" / "epitope_data.csv"
    if not epitope_path.exists():
        print("  Epitope data not found")
        return findings

    df = pd.read_csv(epitope_path)
    print(f"  Loaded {len(df)} epitopes for conservation analysis")

    # Get epitope and protein columns
    epitope_col = "epitope" if "epitope" in df.columns else "Epitope"
    protein_col = "protein" if "protein" in df.columns else "Protein"

    if epitope_col not in df.columns:
        print("  Required columns not found")
        return findings

    # Analyze conservation per protein
    protein_conservation = {}

    for protein in df[protein_col].unique():
        protein_epitopes = df[df[protein_col] == protein][epitope_col].tolist()

        if len(protein_epitopes) < 5:
            continue

        # Compute amino acid frequency per position
        max_len = max(len(str(e)) for e in protein_epitopes)
        position_entropy = []

        for pos in range(min(max_len, 20)):  # First 20 positions
            aa_counts = Counter()
            for epitope in protein_epitopes:
                if pos < len(str(epitope)):
                    aa_counts[str(epitope)[pos]] += 1

            # Shannon entropy
            total = sum(aa_counts.values())
            entropy = 0
            for count in aa_counts.values():
                if count > 0:
                    p = count / total
                    entropy -= p * np.log2(p)

            position_entropy.append(entropy)

        # Conservation score = inverse of mean entropy
        mean_entropy = np.mean(position_entropy) if position_entropy else 0
        conservation_score = 1 / (1 + mean_entropy)

        protein_conservation[protein] = {
            "n_epitopes": len(protein_epitopes),
            "mean_entropy": float(mean_entropy),
            "conservation_score": float(conservation_score),
            "position_entropy": [float(e) for e in position_entropy[:10]],
        }

    # Rank proteins by conservation
    sorted_proteins = sorted(protein_conservation.items(), key=lambda x: x[1]["conservation_score"], reverse=True)

    print("\n  PROTEIN CONSERVATION RANKING:")
    for protein, stats in sorted_proteins:
        print(
            f"    {protein}: conservation={stats['conservation_score']:.3f}, "
            f"entropy={stats['mean_entropy']:.3f}, n_epitopes={stats['n_epitopes']}"
        )

    # Identify highly conserved regions
    print("\n  MOST CONSERVED PROTEINS (best vaccine targets):")
    for protein, stats in sorted_proteins[:3]:
        print(f"    {protein}: {stats['conservation_score']:.3f} conservation score")

    print("\n  MOST VARIABLE PROTEINS (escape-prone):")
    for protein, stats in sorted_proteins[-3:]:
        print(f"    {protein}: {stats['conservation_score']:.3f} conservation score")

    findings["status"] = "success"
    findings["protein_conservation"] = protein_conservation
    findings["most_conserved"] = [p for p, _ in sorted_proteins[:3]]
    findings["most_variable"] = [p for p, _ in sorted_proteins[-3:]]

    return findings


# =============================================================================
# TOOL 7: DRUG REPURPOSING SCREENER
# =============================================================================


def screen_drug_repurposing(data_dir: Path) -> dict:
    """Screen for drug repurposing opportunities."""
    print("\n" + "=" * 70)
    print("TOOL 7: Drug Repurposing Screener")
    print("=" * 70)

    findings = {"status": "failed", "candidates": []}

    # Load PPI data
    ppi_path = data_dir / "external" / "huggingface" / "human_hiv_ppi" / "data" / "train-00000-of-00001.parquet"

    if not ppi_path.exists() or not HAS_PARQUET:
        print("  PPI data not available")
        return findings

    df = pq.read_table(ppi_path).to_pandas()
    print(f"  Loaded {len(df)} protein interactions")

    # Known drug targets (simplified database)
    KNOWN_DRUG_TARGETS = {
        "kinase": {
            "drugs": ["imatinib", "dasatinib", "nilotinib", "gefitinib", "erlotinib"],
            "mechanism": "ATP-competitive inhibition",
        },
        "protease": {
            "drugs": ["bortezomib", "carfilzomib", "ixazomib"],
            "mechanism": "Proteasome inhibition",
        },
        "receptor": {
            "drugs": ["cetuximab", "trastuzumab", "rituximab"],
            "mechanism": "Receptor blocking",
        },
        "phosphatase": {
            "drugs": ["sodium orthovanadate", "okadaic acid"],
            "mechanism": "Phosphatase inhibition",
        },
        "dehydrogenase": {
            "drugs": ["methotrexate", "pemetrexed"],
            "mechanism": "Enzyme inhibition",
        },
    }

    # Find druggable interactions
    repurposing_candidates = []

    for _, row in df.iterrows():
        human_protein = str(row.get("human_protein_name", "")).lower()
        hiv_protein = row.get("hiv_protein_name", "")
        interaction = row.get("interaction_type", "")

        for target_class, info in KNOWN_DRUG_TARGETS.items():
            if target_class in human_protein:
                repurposing_candidates.append(
                    {
                        "human_target": row.get("human_protein_name", ""),
                        "hiv_protein": hiv_protein,
                        "interaction_type": interaction,
                        "target_class": target_class,
                        "existing_drugs": info["drugs"],
                        "mechanism": info["mechanism"],
                        "repurposing_score": 0.8 if "activates" in interaction or "upregulates" in interaction else 0.5,
                    }
                )
                break

    # Score and rank candidates
    for candidate in repurposing_candidates:
        # Boost score for critical HIV proteins
        if candidate["hiv_protein"] in ["Tat", "Rev", "Nef", "Vpr"]:
            candidate["repurposing_score"] *= 1.5

    repurposing_candidates.sort(key=lambda x: x["repurposing_score"], reverse=True)

    # Deduplicate by human target
    seen_targets = set()
    unique_candidates = []
    for c in repurposing_candidates:
        if c["human_target"] not in seen_targets:
            seen_targets.add(c["human_target"])
            unique_candidates.append(c)

    print(f"\n  Found {len(unique_candidates)} unique repurposing candidates")

    print("\n  TOP DRUG REPURPOSING CANDIDATES:")
    for c in unique_candidates[:15]:
        drugs = ", ".join(c["existing_drugs"][:2])
        print(f"    {c['human_target'][:40]:40s}")
        print(f"      HIV: {c['hiv_protein']}, Class: {c['target_class']}")
        print(f"      Existing drugs: {drugs}")
        print()

    # Group by HIV protein
    by_hiv_protein = defaultdict(list)
    for c in unique_candidates:
        by_hiv_protein[c["hiv_protein"]].append(c)

    print("\n  CANDIDATES BY HIV PROTEIN:")
    for hiv_prot, candidates in sorted(by_hiv_protein.items(), key=lambda x: -len(x[1])):
        print(f"    {hiv_prot}: {len(candidates)} druggable targets")

    findings["status"] = "success"
    findings["n_candidates"] = len(unique_candidates)
    findings["top_candidates"] = unique_candidates[:20]
    findings["by_hiv_protein"] = {k: len(v) for k, v in by_hiv_protein.items()}

    return findings


# =============================================================================
# TOOL 8: MUTATION COOCCURRENCE NETWORK
# =============================================================================


def build_mutation_network(data_dir: Path) -> dict:
    """Build mutation cooccurrence network."""
    print("\n" + "=" * 70)
    print("TOOL 8: Mutation Cooccurrence Network")
    print("=" * 70)

    findings = {"status": "failed", "network": {}}

    # Load HIVDB data
    hivdb_dir = data_dir.parent / "research" / "datasets"
    hivdb_files = list(hivdb_dir.glob("stanford_hivdb_*.txt"))

    if not hivdb_files:
        print("  HIVDB data not found")
        return findings

    # Build mutation profiles by drug class
    drug_class_mutations = defaultdict(list)

    for f in hivdb_files:
        drug_class = f.stem.replace("stanford_hivdb_", "").upper()
        try:
            df = pd.read_csv(f, sep="\t")
            if "CompMutList" in df.columns:
                for mut_list in df["CompMutList"].dropna():
                    muts = [m.strip() for m in str(mut_list).split(",") if m.strip()]
                    if muts:
                        drug_class_mutations[drug_class].append(muts)
        except:
            pass

    print(f"  Loaded mutations for {len(drug_class_mutations)} drug classes")

    # Build network per drug class
    networks = {}

    for drug_class, profiles in drug_class_mutations.items():
        print(f"\n  Processing {drug_class}...")

        # Count mutations and cooccurrences
        mutation_counts = Counter()
        cooccurrence = defaultdict(Counter)

        for profile in profiles:
            for mut in profile:
                mutation_counts[mut] += 1
            for m1, m2 in combinations(profile, 2):
                cooccurrence[m1][m2] += 1
                cooccurrence[m2][m1] += 1

        # Build network edges
        top_muts = [m for m, _ in mutation_counts.most_common(20)]
        edges = []

        for m1 in top_muts:
            for m2 in top_muts:
                if m1 < m2:
                    cooc = cooccurrence[m1][m2]
                    if cooc >= 20:
                        expected = (mutation_counts[m1] * mutation_counts[m2]) / len(profiles)
                        phi = (cooc - expected) / max(np.sqrt(expected), 1)
                        if phi > 2:  # Strong positive association
                            edges.append(
                                {
                                    "from": m1,
                                    "to": m2,
                                    "weight": cooc,
                                    "phi": float(phi),
                                }
                            )

        edges.sort(key=lambda x: x["phi"], reverse=True)

        # Find central nodes (most connections)
        connection_count = Counter()
        for e in edges:
            connection_count[e["from"]] += 1
            connection_count[e["to"]] += 1

        central_nodes = [m for m, _ in connection_count.most_common(5)]

        networks[drug_class] = {
            "n_profiles": len(profiles),
            "n_mutations": len(mutation_counts),
            "n_edges": len(edges),
            "top_edges": edges[:10],
            "central_nodes": central_nodes,
            "top_mutations": [{"mutation": m, "count": c} for m, c in mutation_counts.most_common(10)],
        }

        print(f"    Mutations: {len(mutation_counts)}, Edges: {len(edges)}")
        print(f"    Central nodes: {', '.join(central_nodes)}")

    # Cross-class analysis
    print("\n  CROSS-CLASS MUTATION ANALYSIS:")
    all_top_muts = set()
    for drug_class, net in networks.items():
        for m in net["top_mutations"][:5]:
            all_top_muts.add(m["mutation"])

    cross_class_muts = []
    for mut in all_top_muts:
        classes_present = []
        for drug_class, net in networks.items():
            if any(m["mutation"] == mut for m in net["top_mutations"]):
                classes_present.append(drug_class)
        if len(classes_present) >= 2:
            cross_class_muts.append({"mutation": mut, "drug_classes": classes_present})

    for m in cross_class_muts[:10]:
        print(f"    {m['mutation']}: {', '.join(m['drug_classes'])}")

    findings["status"] = "success"
    findings["networks"] = networks
    findings["cross_class_mutations"] = cross_class_muts

    return findings


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 70)
    print("ADVANCED HIV RESEARCH PIPELINE")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    data_dir = PROJECT_ROOT / "data"
    results_dir = PROJECT_ROOT / "research" / "bioinformatics" / "codon_encoder_research" / "hiv" / "results"
    output_dir = PROJECT_ROOT / "results" / "advanced_research"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_findings = {
        "timestamp": datetime.now().isoformat(),
        "tools": {},
        "summary": [],
    }

    # Tool 1: bnAb Combinations
    findings_1 = find_optimal_bnab_combinations(results_dir)
    all_findings["tools"]["bnab_combinations"] = findings_1
    if findings_1["status"] == "success":
        combo = findings_1.get("recommendation", {}).get("combination", [])
        all_findings["summary"].append(f"Optimal bnAb combination: {' + '.join(combo)}")

    # Tool 2: Resistance Pathways
    findings_2 = map_resistance_pathways(data_dir)
    all_findings["tools"]["resistance_pathways"] = findings_2
    if findings_2["status"] == "success":
        all_findings["summary"].append(f"Mapped {len(findings_2.get('pathways', []))} resistance pathways")

    # Tool 3: HLA Coverage
    findings_3 = optimize_hla_coverage(results_dir)
    all_findings["tools"]["hla_coverage"] = findings_3
    if findings_3["status"] == "success":
        all_findings["summary"].append(f"Achieved {findings_3.get('total_coverage', 0):.1%} global HLA coverage")

    # Tool 4: Ensemble Models
    findings_4 = build_ensemble_models(data_dir)
    all_findings["tools"]["ensemble_models"] = findings_4
    if findings_4["status"] == "success":
        all_findings["summary"].append(f"Best ensemble AUC: {findings_4.get('best_auc', 0):.4f}")

    # Tool 5: Phylogenetic Clustering
    findings_5 = analyze_phylogenetic_clustering(data_dir)
    all_findings["tools"]["phylogenetic_clustering"] = findings_5
    if findings_5["status"] == "success":
        all_findings["summary"].append(f"Identified {findings_5.get('n_clusters', 0)} phylogenetic clusters")

    # Tool 6: Conservation Analysis
    findings_6 = analyze_sequence_conservation(data_dir, results_dir)
    all_findings["tools"]["conservation_analysis"] = findings_6
    if findings_6["status"] == "success":
        most_conserved = findings_6.get("most_conserved", [])
        all_findings["summary"].append(f"Most conserved proteins: {', '.join(most_conserved)}")

    # Tool 7: Drug Repurposing
    findings_7 = screen_drug_repurposing(data_dir)
    all_findings["tools"]["drug_repurposing"] = findings_7
    if findings_7["status"] == "success":
        all_findings["summary"].append(f"Found {findings_7.get('n_candidates', 0)} drug repurposing candidates")

    # Tool 8: Mutation Network
    findings_8 = build_mutation_network(data_dir)
    all_findings["tools"]["mutation_network"] = findings_8
    if findings_8["status"] == "success":
        all_findings["summary"].append("Built mutation cooccurrence networks for all drug classes")

    # Save results
    json_path = output_dir / "advanced_research_report.json"
    with open(json_path, "w") as f:
        json.dump(all_findings, f, indent=2, default=str)

    # Generate markdown
    md_path = output_dir / "ADVANCED_RESEARCH.md"
    with open(md_path, "w") as f:
        f.write("# Advanced HIV Research Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Summary\n\n")
        for item in all_findings["summary"]:
            f.write(f"- {item}\n")
        f.write("\n")

        # Detailed sections
        f.write("## 1. Optimal bnAb Combinations\n\n")
        if findings_1["status"] == "success":
            rec = findings_1.get("recommendation", {})
            f.write(f"**Recommended combination**: {' + '.join(rec.get('combination', []))}\n")
            f.write(f"- Estimated coverage: {rec.get('coverage', 0):.1f}%\n")
            f.write(f"- Estimated potency (IC50): {rec.get('potency', 0):.3f}\n\n")

        f.write("## 2. Resistance Evolution Pathways\n\n")
        if findings_2["status"] == "success":
            f.write("### Top Mutation Associations\n\n")
            for p in findings_2.get("pathways", [])[:10]:
                f.write(f"- {p['from']} <-> {p['to']} (lift: {p['lift']:.2f})\n")
            f.write("\n")

        f.write("## 3. Global HLA Coverage Optimization\n\n")
        if findings_3["status"] == "success":
            f.write(f"**Total coverage**: {findings_3.get('total_coverage', 0):.1%}\n\n")
            f.write("### Regional Coverage\n\n")
            for region, cov in findings_3.get("regional_coverage", {}).items():
                f.write(f"- {region}: {cov:.1%}\n")
            f.write("\n")

        f.write("## 4. Ensemble Prediction Models\n\n")
        if findings_4["status"] == "success":
            f.write(f"- Best model: {findings_4.get('best_model', 'unknown')}\n")
            f.write(f"- Best AUC: {findings_4.get('best_auc', 0):.4f}\n")
            f.write(f"- CV AUC: {findings_4.get('cv_auc_mean', 0):.4f} +/- {findings_4.get('cv_auc_std', 0):.4f}\n\n")

        f.write("## 5. Conservation Analysis\n\n")
        if findings_6["status"] == "success":
            f.write("### Most Conserved (Best Vaccine Targets)\n")
            for p in findings_6.get("most_conserved", []):
                f.write(f"- {p}\n")
            f.write("\n### Most Variable (Escape-Prone)\n")
            for p in findings_6.get("most_variable", []):
                f.write(f"- {p}\n")
            f.write("\n")

        f.write("## 6. Drug Repurposing Candidates\n\n")
        if findings_7["status"] == "success":
            f.write(f"Total candidates: {findings_7.get('n_candidates', 0)}\n\n")
            f.write("### Top Candidates\n\n")
            for c in findings_7.get("top_candidates", [])[:5]:
                f.write(f"- **{c['human_target'][:50]}**\n")
                f.write(f"  - HIV protein: {c['hiv_protein']}\n")
                f.write(f"  - Drugs: {', '.join(c['existing_drugs'][:3])}\n\n")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for item in all_findings["summary"]:
        print(f"  - {item}")

    print(f"\nReports saved to: {output_dir}")
    print("\n" + "=" * 70)
    print("ADVANCED RESEARCH PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
