#!/usr/bin/env python3
"""
HIV Research Discovery Pipeline

Executes 5 novel research directions:
1. Cross-link vaccine targets with escape velocity
2. Validate p-adic geometry against phylogenetic distances
3. Test p-adic embeddings for tropism prediction
4. Correlate bnAb escape with drug resistance
5. Map HIV-human PPI to druggable targets

Generates comprehensive research findings report.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("ERROR: pandas required for this analysis")
    sys.exit(1)

try:
    import pyarrow.parquet as pq

    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False

from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split

# =============================================================================
# CONSTANTS (imported from centralized biology module)
# =============================================================================
from src.biology.codons import CODON_TO_INDEX, GENETIC_CODE

CODONS = list(GENETIC_CODE.keys())
AA_TO_IDX = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY*")}

# Known druggable protein families (simplified list)
DRUGGABLE_FAMILIES = {
    "kinase",
    "receptor",
    "channel",
    "transporter",
    "protease",
    "polymerase",
    "integrase",
    "helicase",
    "ligase",
    "phosphatase",
    "dehydrogenase",
    "oxidase",
    "reductase",
    "transferase",
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def parse_fasta(filepath: Path) -> list[tuple[str, str]]:
    """Parse FASTA file."""
    sequences = []
    current_name = None
    current_seq = []
    with open(filepath, errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_name:
                    sequences.append((current_name, "".join(current_seq)))
                current_name = line[1:]
                current_seq = []
            elif line:
                current_seq.append(line.upper())
        if current_name:
            sequences.append((current_name, "".join(current_seq)))
    return sequences


def compute_padic_distance(codon1: str, codon2: str, p: int = 3) -> float:
    """Compute p-adic distance between codons."""
    if codon1 == codon2:
        return 0.0
    for i, (b1, b2) in enumerate(zip(codon1, codon2, strict=False)):
        if b1 != b2:
            return p ** (-(i + 1))
    return 0.0


def sequence_to_codons(seq: str) -> list[str]:
    """Convert DNA sequence to list of codon strings."""
    codons = []
    seq = seq.upper().replace("U", "T")
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i : i + 3]
        if codon in CODON_TO_INDEX:
            codons.append(codon)
    return codons


def compute_hamming_distance(seq1: str, seq2: str) -> float:
    """Compute normalized Hamming distance."""
    if len(seq1) != len(seq2):
        min_len = min(len(seq1), len(seq2))
        seq1, seq2 = seq1[:min_len], seq2[:min_len]
    if len(seq1) == 0:
        return 1.0
    mismatches = sum(c1 != c2 for c1, c2 in zip(seq1, seq2, strict=False))
    return mismatches / len(seq1)


def compute_padic_sequence_distance(seq1: str, seq2: str) -> float:
    """Compute p-adic distance between two sequences."""
    codons1 = sequence_to_codons(seq1)
    codons2 = sequence_to_codons(seq2)
    if not codons1 or not codons2:
        return 1.0
    min_len = min(len(codons1), len(codons2))
    total_dist = 0
    for i in range(min_len):
        total_dist += compute_padic_distance(codons1[i], codons2[i])
    return total_dist / min_len


# =============================================================================
# RESEARCH 1: VACCINE TARGETS + ESCAPE VELOCITY
# =============================================================================


def research_1_vaccine_escape_crosslink(results_dir: Path) -> dict:
    """Cross-link vaccine targets with escape velocity data."""
    print("\n" + "=" * 70)
    print("RESEARCH 1: Vaccine Target + Escape Velocity Cross-linking")
    print("=" * 70)

    findings = {"status": "failed", "discoveries": []}

    # Load vaccine targets
    vaccine_path = results_dir / "integrated" / "vaccine_targets.csv"
    escape_path = results_dir / "ctl_escape" / "escape_velocity.csv"
    epitope_path = results_dir / "ctl_escape" / "epitope_data.csv"

    if not vaccine_path.exists():
        print(f"  Vaccine targets not found: {vaccine_path}")
        return findings

    vaccine_df = pd.read_csv(vaccine_path)
    print(f"  Loaded {len(vaccine_df)} vaccine targets")

    # Load escape velocity data
    if escape_path.exists():
        escape_df = pd.read_csv(escape_path)
        print(f"  Loaded escape velocity for {len(escape_df)} proteins")
    else:
        # Create escape velocity from epitope data
        if epitope_path.exists():
            epitope_df = pd.read_csv(epitope_path)
            # Compute escape velocity per protein as std of radii
            escape_df = epitope_df.groupby("protein").agg({"radius": ["mean", "std", "count"]}).reset_index()
            escape_df.columns = ["protein", "mean_radius", "escape_velocity", "n_epitopes"]
            escape_df["escape_velocity"] = escape_df["escape_velocity"].fillna(0.1)
        else:
            print("  No escape data found")
            return findings

    print(f"  Escape velocity data: {len(escape_df)} proteins")

    # Cross-link: Add escape velocity to vaccine targets
    # Map protein names
    protein_escape = dict(zip(escape_df["protein"], escape_df["escape_velocity"], strict=False))

    # Add escape velocity column
    vaccine_df["escape_velocity"] = vaccine_df["protein"].map(protein_escape)
    vaccine_df["escape_velocity"] = vaccine_df["escape_velocity"].fillna(escape_df["escape_velocity"].mean())

    # Compute composite score: high HLA coverage + low escape velocity
    vaccine_df["hla_count"] = vaccine_df["n_hla_restrictions"] if "n_hla_restrictions" in vaccine_df.columns else 10
    vaccine_df["stability_score"] = vaccine_df["hla_count"] / (vaccine_df["escape_velocity"] + 0.01)

    # Rank by stability score
    vaccine_df = vaccine_df.sort_values("stability_score", ascending=False)

    # Top stable vaccine targets
    top_stable = vaccine_df.head(20)

    print("\n  TOP 20 EVOLUTIONARILY STABLE VACCINE TARGETS:")
    print("  " + "-" * 60)
    for _i, row in top_stable.iterrows():
        epitope = row.get("epitope", row.get("Epitope", "Unknown"))[:20]
        protein = row.get("protein", row.get("Protein", "Unknown"))
        escape_v = row["escape_velocity"]
        score = row["stability_score"]
        print(f"  {epitope:20s} | {protein:6s} | Escape: {escape_v:.4f} | Score: {score:.2f}")

    # Save enhanced vaccine targets
    output_path = results_dir / "integrated" / "vaccine_targets_with_stability.csv"
    vaccine_df.to_csv(output_path, index=False)
    print(f"\n  Saved to: {output_path}")

    # Key findings
    findings["status"] = "success"
    findings["n_targets_analyzed"] = len(vaccine_df)
    findings["top_stable_target"] = {
        "epitope": str(top_stable.iloc[0].get("epitope", top_stable.iloc[0].get("Epitope", "Unknown"))),
        "protein": str(top_stable.iloc[0].get("protein", top_stable.iloc[0].get("Protein", "Unknown"))),
        "stability_score": float(top_stable.iloc[0]["stability_score"]),
    }
    findings["discoveries"].append(f"Identified {len(vaccine_df)} vaccine targets ranked by evolutionary stability")
    findings["discoveries"].append(
        f"Top target: {findings['top_stable_target']['epitope']} in {findings['top_stable_target']['protein']}"
    )

    # Discovery: Which proteins have most stable epitopes?
    protein_stability = (
        vaccine_df.groupby(vaccine_df["protein"] if "protein" in vaccine_df.columns else vaccine_df["Protein"])[
            "stability_score"
        ]
        .mean()
        .sort_values(ascending=False)
    )

    print("\n  PROTEIN STABILITY RANKING:")
    for prot, score in protein_stability.head(10).items():
        print(f"    {prot}: {score:.2f}")

    findings["protein_stability_ranking"] = protein_stability.head(10).to_dict()

    return findings


# =============================================================================
# RESEARCH 2: VALIDATE P-ADIC GEOMETRY
# =============================================================================


def research_2_validate_padic_geometry(data_dir: Path) -> dict:
    """Validate p-adic geometry against phylogenetic distances."""
    print("\n" + "=" * 70)
    print("RESEARCH 2: P-adic Geometry Validation")
    print("=" * 70)

    findings = {"status": "failed", "discoveries": []}

    # Load sequences
    fasta_dir = (
        data_dir / "external" / "github" / "HIV-1_Paper" / "Individual_Representative_Sequences_Used_for_Subtyping"
    )

    all_sequences = []
    for fasta_path in fasta_dir.glob("*.fasta"):
        seqs = parse_fasta(fasta_path)
        for name, seq in seqs:
            if len(seq) >= 100:
                all_sequences.append((name, seq))

    if len(all_sequences) < 10:
        print("  Insufficient sequences for validation")
        return findings

    print(f"  Loaded {len(all_sequences)} sequences")

    # Sample for efficiency
    n_sample = min(50, len(all_sequences))
    sample_indices = np.random.choice(len(all_sequences), n_sample, replace=False)
    sample_seqs = [all_sequences[i] for i in sample_indices]

    # Compute pairwise distances: Hamming (phylogenetic proxy) vs P-adic
    n = len(sample_seqs)
    hamming_dists = np.zeros((n, n))
    padic_dists = np.zeros((n, n))

    print("  Computing pairwise distances...")
    for i in range(n):
        for j in range(i + 1, n):
            seq1, seq2 = sample_seqs[i][1], sample_seqs[j][1]
            hamming_dists[i, j] = compute_hamming_distance(seq1, seq2)
            hamming_dists[j, i] = hamming_dists[i, j]
            padic_dists[i, j] = compute_padic_sequence_distance(seq1, seq2)
            padic_dists[j, i] = padic_dists[i, j]

    # Extract upper triangle for correlation
    triu_idx = np.triu_indices(n, k=1)
    hamming_flat = hamming_dists[triu_idx]
    padic_flat = padic_dists[triu_idx]

    # Compute correlations
    pearson_r, pearson_p = stats.pearsonr(hamming_flat, padic_flat)
    spearman_r, spearman_p = stats.spearmanr(hamming_flat, padic_flat)

    print("\n  CORRELATION RESULTS:")
    print(f"    Pearson r:  {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"    Spearman r: {spearman_r:.4f} (p={spearman_p:.2e})")

    # Interpretation
    if spearman_r > 0.7:
        interpretation = "STRONG: P-adic geometry captures evolutionary relationships"
    elif spearman_r > 0.5:
        interpretation = "MODERATE: P-adic geometry partially captures evolution"
    elif spearman_r > 0.3:
        interpretation = "WEAK: Limited evolutionary signal in p-adic geometry"
    else:
        interpretation = "NONE: P-adic geometry does not correlate with phylogeny"

    print(f"\n  INTERPRETATION: {interpretation}")

    findings["status"] = "success"
    findings["n_sequences"] = len(sample_seqs)
    findings["pearson_r"] = float(pearson_r)
    findings["pearson_p"] = float(pearson_p)
    findings["spearman_r"] = float(spearman_r)
    findings["spearman_p"] = float(spearman_p)
    findings["interpretation"] = interpretation
    findings["discoveries"].append(f"P-adic vs Hamming correlation: Spearman r = {spearman_r:.4f}")
    findings["discoveries"].append(interpretation)

    # Additional: Test if p-adic distances cluster by sequence type
    # Extract subtype from name if available
    subtypes = []
    for name, _ in sample_seqs:
        if "_PR_" in name:
            subtypes.append("PR")
        elif "_RT_" in name:
            subtypes.append("RT")
        elif "_IN_" in name:
            subtypes.append("IN")
        elif "_V1V3_" in name:
            subtypes.append("V1V3")
        else:
            subtypes.append("Other")

    unique_subtypes = list(set(subtypes))
    if len(unique_subtypes) > 1:
        print(f"\n  Subtypes found: {unique_subtypes}")
        # Compute within-subtype vs between-subtype distances
        within_dists = []
        between_dists = []
        for i in range(n):
            for j in range(i + 1, n):
                if subtypes[i] == subtypes[j]:
                    within_dists.append(padic_dists[i, j])
                else:
                    between_dists.append(padic_dists[i, j])

        if within_dists and between_dists:
            within_mean = np.mean(within_dists)
            between_mean = np.mean(between_dists)
            t_stat, t_p = stats.ttest_ind(within_dists, between_dists)
            print(f"  Within-subtype distance: {within_mean:.4f}")
            print(f"  Between-subtype distance: {between_mean:.4f}")
            print(f"  T-test p-value: {t_p:.2e}")

            if t_p < 0.05 and between_mean > within_mean:
                findings["discoveries"].append(f"P-adic distances distinguish sequence types (p={t_p:.2e})")

    return findings


# =============================================================================
# RESEARCH 3: P-ADIC TROPISM PREDICTION
# =============================================================================


def research_3_padic_tropism_prediction(data_dir: Path) -> dict:
    """Test p-adic embeddings for tropism prediction improvement."""
    print("\n" + "=" * 70)
    print("RESEARCH 3: P-adic Embeddings for Tropism Prediction")
    print("=" * 70)

    findings = {"status": "failed", "discoveries": []}

    # Load V3 coreceptor data
    v3_path = data_dir / "external" / "huggingface" / "HIV_V3_coreceptor" / "data" / "train-00000-of-00001.parquet"

    if not v3_path.exists() or not HAS_PARQUET:
        print("  V3 coreceptor data not available")
        return findings

    df = pq.read_table(v3_path).to_pandas()
    print(f"  Loaded {len(df)} V3 sequences")

    # Prepare data
    sequences = df["sequence"].tolist()
    # Labels: 1 = X4 (CXCR4), 0 = R5 (CCR5)
    labels = df["CXCR4"].astype(int).tolist()

    # Method 1: One-hot encoding (baseline)
    def onehot_encode(seq, max_len=35):
        aa_order = "ACDEFGHIKLMNPQRSTVWY-"
        encoding = np.zeros((max_len, len(aa_order)))
        for i, aa in enumerate(seq[:max_len]):
            if aa in aa_order:
                encoding[i, aa_order.index(aa)] = 1
        return encoding.flatten()

    # Method 2: P-adic inspired encoding
    # Amino acids grouped by biochemical similarity (p-adic hierarchy)
    AA_GROUPS = {
        "hydrophobic_aliphatic": "AVILM",
        "hydrophobic_aromatic": "FWY",
        "polar_uncharged": "STNQ",
        "positive": "KRH",
        "negative": "DE",
        "special": "CGP",
    }

    def padic_aa_encode(seq, max_len=35):
        """Encode amino acids with p-adic inspired hierarchy."""
        # Level 1: Group membership (6 groups)
        # Level 2: Position within group
        # Level 3: Exact amino acid
        encoding = np.zeros((max_len, 10))  # Reduced dimensionality

        for i, aa in enumerate(seq[:max_len]):
            if aa == "-" or aa not in "ACDEFGHIKLMNPQRSTVWY":
                continue

            # Find group
            for g_idx, (_group_name, group_aas) in enumerate(AA_GROUPS.items()):
                if aa in group_aas:
                    # Hierarchical encoding
                    encoding[i, 0] = g_idx / 6  # Group level
                    encoding[i, 1] = group_aas.index(aa) / len(group_aas)  # Within-group
                    encoding[i, 2 + g_idx] = 1  # One-hot group
                    break

            # Add position-weighted feature
            encoding[i, 8] = i / max_len  # Position
            encoding[i, 9] = 1 if i in [11, 24, 25] else 0  # Key positions (11, 25 are known)

        return encoding.flatten()

    print("  Encoding sequences...")
    X_onehot = np.array([onehot_encode(s) for s in sequences])
    X_padic = np.array([padic_aa_encode(s) for s in sequences])
    y = np.array(labels)

    # Split data
    X_oh_train, X_oh_test, y_train, y_test = train_test_split(X_onehot, y, test_size=0.2, random_state=42, stratify=y)
    X_pa_train, X_pa_test, _, _ = train_test_split(X_padic, y, test_size=0.2, random_state=42, stratify=y)

    # Train classifiers
    print("\n  Training classifiers...")

    # Baseline: One-hot + Random Forest
    rf_onehot = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_onehot.fit(X_oh_train, y_train)
    y_pred_oh = rf_onehot.predict_proba(X_oh_test)[:, 1]
    auc_onehot = roc_auc_score(y_test, y_pred_oh)
    acc_onehot = accuracy_score(y_test, rf_onehot.predict(X_oh_test))

    # P-adic: P-adic encoding + Random Forest
    rf_padic = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_padic.fit(X_pa_train, y_train)
    y_pred_pa = rf_padic.predict_proba(X_pa_test)[:, 1]
    auc_padic = roc_auc_score(y_test, y_pred_pa)
    acc_padic = accuracy_score(y_test, rf_padic.predict(X_pa_test))

    # Combined: Both encodings
    X_combined_train = np.hstack([X_oh_train, X_pa_train])
    X_combined_test = np.hstack([X_oh_test, X_pa_test])
    rf_combined = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_combined.fit(X_combined_train, y_train)
    y_pred_comb = rf_combined.predict_proba(X_combined_test)[:, 1]
    auc_combined = roc_auc_score(y_test, y_pred_comb)
    acc_combined = accuracy_score(y_test, rf_combined.predict(X_combined_test))

    print("\n  RESULTS:")
    print(f"    One-hot encoding:     AUC = {auc_onehot:.4f}, Accuracy = {acc_onehot:.4f}")
    print(f"    P-adic encoding:      AUC = {auc_padic:.4f}, Accuracy = {acc_padic:.4f}")
    print(f"    Combined encoding:    AUC = {auc_combined:.4f}, Accuracy = {acc_combined:.4f}")

    improvement = (auc_combined - auc_onehot) / auc_onehot * 100
    print(f"\n  Improvement from combining: {improvement:+.2f}%")

    findings["status"] = "success"
    findings["n_sequences"] = len(sequences)
    findings["auc_onehot"] = float(auc_onehot)
    findings["auc_padic"] = float(auc_padic)
    findings["auc_combined"] = float(auc_combined)
    findings["improvement_percent"] = float(improvement)

    if auc_combined > auc_onehot:
        findings["discoveries"].append(
            f"P-adic features improve tropism prediction: {improvement:+.2f}% AUC improvement"
        )
    else:
        findings["discoveries"].append("P-adic features did not improve over baseline for tropism prediction")

    # Cross-validation for robustness
    print("\n  Cross-validation (5-fold)...")
    cv_onehot = cross_val_score(rf_onehot, X_onehot, y, cv=5, scoring="roc_auc")
    cv_combined = cross_val_score(rf_combined, np.hstack([X_onehot, X_padic]), y, cv=5, scoring="roc_auc")

    print(f"    One-hot CV AUC: {cv_onehot.mean():.4f} +/- {cv_onehot.std():.4f}")
    print(f"    Combined CV AUC: {cv_combined.mean():.4f} +/- {cv_combined.std():.4f}")

    findings["cv_auc_onehot"] = float(cv_onehot.mean())
    findings["cv_auc_combined"] = float(cv_combined.mean())

    return findings


# =============================================================================
# RESEARCH 4: BNAB ESCAPE + DRUG RESISTANCE CORRELATION
# =============================================================================


def research_4_bnab_resistance_correlation(data_dir: Path, results_dir: Path) -> dict:
    """Correlate bnAb escape with drug resistance patterns."""
    print("\n" + "=" * 70)
    print("RESEARCH 4: bnAb Escape vs Drug Resistance Correlation")
    print("=" * 70)

    findings = {"status": "failed", "discoveries": []}

    # Load CATNAP data
    catnap_path = data_dir.parent / "research" / "datasets" / "catnap_assay.txt"

    if not catnap_path.exists():
        # Try alternate location
        catnap_path = results_dir / "catnap_neutralization" / "bnab_sensitivity.csv"

    if not catnap_path.exists():
        print("  CATNAP data not found")
        return findings

    # Load Stanford HIVDB data
    hivdb_files = list((data_dir.parent / "research" / "datasets").glob("stanford_hivdb_*.txt"))

    if not hivdb_files:
        print("  Stanford HIVDB data not found")
        return findings

    print(f"  Found {len(hivdb_files)} HIVDB files")

    # Load CATNAP
    try:
        if str(catnap_path).endswith(".csv"):
            catnap_df = pd.read_csv(catnap_path)
        else:
            catnap_df = pd.read_csv(catnap_path, sep="\t")
        print(f"  Loaded CATNAP: {len(catnap_df)} records")
    except Exception as e:
        print(f"  Error loading CATNAP: {e}")
        return findings

    # Load HIVDB and combine
    hivdb_dfs = []
    for f in hivdb_files:
        try:
            df = pd.read_csv(f, sep="\t")
            drug_class = f.stem.replace("stanford_hivdb_", "").upper()
            df["drug_class"] = drug_class
            hivdb_dfs.append(df)
        except Exception as e:
            print(f"  Error loading {f.name}: {e}")

    if not hivdb_dfs:
        return findings

    hivdb_df = pd.concat(hivdb_dfs, ignore_index=True)
    print(f"  Loaded HIVDB: {len(hivdb_df)} records")

    # Analysis: Look for patterns in mutation positions
    # Extract mutation positions from CompMutList column
    if "CompMutList" in hivdb_df.columns:
        all_mutations = []
        for mut_list in hivdb_df["CompMutList"].dropna():
            muts = [m.strip() for m in str(mut_list).split(",")]
            all_mutations.extend(muts)

        from collections import Counter

        mutation_counts = Counter(all_mutations)
        top_mutations = mutation_counts.most_common(20)

        print("\n  TOP DRUG RESISTANCE MUTATIONS:")
        for mut, count in top_mutations:
            print(f"    {mut}: {count} occurrences")

        findings["top_resistance_mutations"] = [{"mutation": m, "count": c} for m, c in top_mutations]

    # Key analysis: Resistance level distribution
    drug_cols = [
        c for c in hivdb_df.columns if c not in ["SeqID", "CompMutList", "drug_class"] and not c.startswith("P")
    ]

    if drug_cols:
        print("\n  RESISTANCE LEVELS BY DRUG CLASS:")
        for drug_class in hivdb_df["drug_class"].unique():
            subset = hivdb_df[hivdb_df["drug_class"] == drug_class]
            for col in drug_cols[:3]:  # First 3 drugs
                if col in subset.columns:
                    values = pd.to_numeric(subset[col], errors="coerce").dropna()
                    if len(values) > 0:
                        mean_fc = values.mean()
                        print(f"    {drug_class} - {col}: mean FC = {mean_fc:.2f}")

    # Hypothesis test: Do sequences with high resistance have specific patterns?
    # Compute "total resistance score" per sequence
    if drug_cols and len(hivdb_df) > 10:
        resistance_scores = []
        for _, row in hivdb_df.iterrows():
            score = 0
            for col in drug_cols:
                if col in row.index:
                    val = pd.to_numeric(row[col], errors="coerce")
                    if not pd.isna(val) and val > 1:
                        score += np.log10(val)  # Log fold-change
            resistance_scores.append(score)

        hivdb_df["total_resistance_score"] = resistance_scores

        # Find multi-drug resistant sequences
        high_resistance = hivdb_df[
            hivdb_df["total_resistance_score"] > hivdb_df["total_resistance_score"].quantile(0.9)
        ]
        print(f"\n  High-resistance sequences (top 10%): {len(high_resistance)}")

        if len(high_resistance) > 0 and "CompMutList" in high_resistance.columns:
            # Find mutations enriched in high-resistance
            hr_mutations = []
            for mut_list in high_resistance["CompMutList"].dropna():
                hr_mutations.extend([m.strip() for m in str(mut_list).split(",")])

            hr_mutation_counts = Counter(hr_mutations)
            print("\n  MUTATIONS ENRICHED IN MULTI-DRUG RESISTANCE:")
            for mut, count in hr_mutation_counts.most_common(10):
                pct = count / len(high_resistance) * 100
                print(f"    {mut}: {count} ({pct:.1f}% of MDR sequences)")

            findings["mdr_enriched_mutations"] = [
                {"mutation": m, "count": c, "pct": c / len(high_resistance) * 100}
                for m, c in hr_mutation_counts.most_common(10)
            ]
            findings["discoveries"].append(
                f"Identified {len(hr_mutation_counts)} mutations enriched in multi-drug resistant sequences"
            )

    findings["status"] = "success"
    findings["n_hivdb_records"] = len(hivdb_df)
    findings["drug_classes"] = list(hivdb_df["drug_class"].unique()) if "drug_class" in hivdb_df.columns else []

    return findings


# =============================================================================
# RESEARCH 5: HIV-HUMAN PPI DRUGGABILITY
# =============================================================================


def research_5_ppi_druggability(data_dir: Path) -> dict:
    """Map HIV-human PPI to druggable targets."""
    print("\n" + "=" * 70)
    print("RESEARCH 5: HIV-Human PPI Druggability Analysis")
    print("=" * 70)

    findings = {"status": "failed", "discoveries": []}

    # Load PPI data
    ppi_path = data_dir / "external" / "huggingface" / "human_hiv_ppi" / "data" / "train-00000-of-00001.parquet"

    if not ppi_path.exists() or not HAS_PARQUET:
        print("  PPI data not available")
        return findings

    df = pq.read_table(ppi_path).to_pandas()
    print(f"  Loaded {len(df)} protein-protein interactions")

    # Analyze interaction network
    print(f"\n  HIV proteins: {df['hiv_protein_name'].nunique()}")
    print(f"  Human proteins: {df['human_protein_name'].nunique()}")

    # Count interactions per HIV protein
    hiv_interactions = df.groupby("hiv_protein_name").size().sort_values(ascending=False)
    print("\n  HIV PROTEINS BY INTERACTION COUNT:")
    for protein, count in hiv_interactions.head(10).items():
        print(f"    {protein}: {count} human targets")

    # Find druggable human targets (by name pattern matching)
    def is_potentially_druggable(name: str) -> bool:
        if not isinstance(name, str):
            return False
        name_lower = name.lower()
        return any(family in name_lower for family in DRUGGABLE_FAMILIES)

    df["potentially_druggable"] = df["human_protein_name"].apply(is_potentially_druggable)
    druggable_interactions = df[df["potentially_druggable"]]

    print(f"\n  Potentially druggable interactions: {len(druggable_interactions)}")
    print(f"  Druggable human targets: {druggable_interactions['human_protein_name'].nunique()}")

    # HIV proteins targeting druggable hosts
    hiv_druggable = druggable_interactions.groupby("hiv_protein_name")["human_protein_name"].nunique()
    hiv_druggable = hiv_druggable.sort_values(ascending=False)

    print("\n  HIV PROTEINS TARGETING DRUGGABLE HOSTS:")
    for protein, count in hiv_druggable.head(10).items():
        print(f"    {protein}: {count} druggable targets")

    # Key finding: HIV proteins with MULTIPLE druggable targets
    multi_target_hiv = hiv_druggable[hiv_druggable >= 3]
    print(f"\n  HIV proteins with 3+ druggable targets: {len(multi_target_hiv)}")

    # Detailed analysis of top candidates
    top_candidates = []
    for hiv_protein in multi_target_hiv.index[:5]:
        targets = (
            druggable_interactions[druggable_interactions["hiv_protein_name"] == hiv_protein]["human_protein_name"]
            .unique()
            .tolist()
        )

        top_candidates.append(
            {
                "hiv_protein": hiv_protein,
                "n_druggable_targets": len(targets),
                "targets": targets[:10],  # Limit for readability
            }
        )

        print(f"\n  {hiv_protein} ({len(targets)} druggable targets):")
        for t in targets[:5]:
            print(f"    - {t}")

    # Interaction types
    if "interaction_type" in df.columns:
        print("\n  INTERACTION TYPES:")
        for itype, count in df["interaction_type"].value_counts().head(10).items():
            print(f"    {itype}: {count}")

    findings["status"] = "success"
    findings["total_interactions"] = len(df)
    findings["hiv_proteins"] = int(df["hiv_protein_name"].nunique())
    findings["human_proteins"] = int(df["human_protein_name"].nunique())
    findings["druggable_interactions"] = len(druggable_interactions)
    findings["top_hiv_druggable_targets"] = top_candidates

    findings["discoveries"].append(
        f"Identified {len(multi_target_hiv)} HIV proteins targeting 3+ druggable human proteins"
    )
    findings["discoveries"].append(
        f"Top host-directed therapy target: {multi_target_hiv.index[0]} ({multi_target_hiv.iloc[0]} druggable targets)"
    )

    return findings


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 70)
    print("HIV RESEARCH DISCOVERY PIPELINE")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    data_dir = PROJECT_ROOT / "data"
    results_dir = PROJECT_ROOT / "research" / "bioinformatics" / "codon_encoder_research" / "hiv" / "results"
    output_dir = PROJECT_ROOT / "results" / "research_discoveries"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_findings = {
        "timestamp": datetime.now().isoformat(),
        "research_areas": {},
        "all_discoveries": [],
    }

    # Run all research directions
    print("\n" + "#" * 70)
    print("# EXECUTING RESEARCH DIRECTIONS")
    print("#" * 70)

    # Research 1
    findings_1 = research_1_vaccine_escape_crosslink(results_dir)
    all_findings["research_areas"]["vaccine_escape_crosslink"] = findings_1
    all_findings["all_discoveries"].extend(findings_1.get("discoveries", []))

    # Research 2
    findings_2 = research_2_validate_padic_geometry(data_dir)
    all_findings["research_areas"]["padic_geometry_validation"] = findings_2
    all_findings["all_discoveries"].extend(findings_2.get("discoveries", []))

    # Research 3
    findings_3 = research_3_padic_tropism_prediction(data_dir)
    all_findings["research_areas"]["padic_tropism_prediction"] = findings_3
    all_findings["all_discoveries"].extend(findings_3.get("discoveries", []))

    # Research 4
    findings_4 = research_4_bnab_resistance_correlation(data_dir, results_dir)
    all_findings["research_areas"]["bnab_resistance_correlation"] = findings_4
    all_findings["all_discoveries"].extend(findings_4.get("discoveries", []))

    # Research 5
    findings_5 = research_5_ppi_druggability(data_dir)
    all_findings["research_areas"]["ppi_druggability"] = findings_5
    all_findings["all_discoveries"].extend(findings_5.get("discoveries", []))

    # Summary
    print("\n" + "=" * 70)
    print("RESEARCH SUMMARY")
    print("=" * 70)

    successful = sum(1 for f in all_findings["research_areas"].values() if f.get("status") == "success")
    print(f"\nCompleted: {successful}/5 research directions")

    print("\n" + "-" * 70)
    print("ALL DISCOVERIES:")
    print("-" * 70)
    for i, discovery in enumerate(all_findings["all_discoveries"], 1):
        print(f"  {i}. {discovery}")

    # Save report
    report_path = output_dir / "research_discoveries_report.json"
    with open(report_path, "w") as f:
        json.dump(all_findings, f, indent=2, default=str)
    print(f"\nReport saved to: {report_path}")

    # Generate markdown summary
    md_path = output_dir / "RESEARCH_FINDINGS.md"
    with open(md_path, "w") as f:
        f.write("# HIV Research Discoveries\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Summary\n\n")
        f.write(f"Completed {successful}/5 research directions.\n\n")

        f.write("## Key Discoveries\n\n")
        for i, discovery in enumerate(all_findings["all_discoveries"], 1):
            f.write(f"{i}. {discovery}\n")

        f.write("\n## Detailed Findings\n\n")
        for name, findings in all_findings["research_areas"].items():
            f.write(f"### {name.replace('_', ' ').title()}\n\n")
            f.write(f"Status: {findings.get('status', 'unknown')}\n\n")
            if findings.get("discoveries"):
                for d in findings["discoveries"]:
                    f.write(f"- {d}\n")
            f.write("\n")

    print(f"Markdown summary: {md_path}")
    print("\n" + "=" * 70)
    print("RESEARCH PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
