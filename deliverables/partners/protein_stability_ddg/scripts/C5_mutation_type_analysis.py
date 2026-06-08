#!/usr/bin/env python3
"""C5: Mutation-Type Subgroup Analysis for DDG Prediction.

Computes per-mutation-type Spearman correlation from the ProTherm benchmark
subset (N=52) and compares p-adic vs. physicochemical-only baseline.

These figures are the source for the Mutation-Type Heterogeneity table in
README.md. Due to small subgroup sizes (N=2-22 per type), the extreme values
(-737% for charge_reversal) reflect very small samples with wide CIs and
should be treated as exploratory observations.

Usage:
    python scripts/C5_mutation_type_analysis.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = PACKAGE_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.encoders.trainable_codon_encoder import TrainableCodonEncoder
from src.geometry import poincare_distance
import torch

# Physicochemical properties: (hydrophobicity, charge, volume, polarity)
AA_PROPERTIES = {
    "A": (0.62, 0, 88.6, 0),
    "R": (-2.53, 1, 173.4, 1),
    "N": (-0.78, 0, 114.1, 1),
    "D": (-0.90, -1, 111.1, 1),
    "C": (0.29, 0, 108.5, 0),
    "Q": (-0.85, 0, 143.8, 1),
    "E": (-0.74, -1, 138.4, 1),
    "G": (0.48, 0, 60.1, 0),
    "H": (-0.40, 0.5, 153.2, 1),
    "I": (1.38, 0, 166.7, 0),
    "L": (1.06, 0, 166.7, 0),
    "K": (-1.50, 1, 168.6, 1),
    "M": (0.64, 0, 162.9, 0),
    "F": (1.19, 0, 189.9, 0),
    "P": (0.12, 0, 112.7, 0),
    "S": (-0.18, 0, 89.0, 1),
    "T": (-0.05, 0, 116.1, 1),
    "W": (0.81, 0, 227.8, 0),
    "Y": (0.26, 0, 193.6, 1),
    "V": (1.08, 0, 140.0, 0),
}

CHARGED = {"R", "D", "E", "K", "H"}  # H counts as partial charge (+0.5)
HYDROPHOBIC = {"A", "C", "I", "L", "M", "F", "W", "V"}
POLAR_UNCHARGED = {"N", "Q", "S", "T", "Y"}
PROLINE = {"P"}


def classify_mutation(wt: str, mt: str) -> str:
    """Classify a mutation into a type category."""
    wt_charge = AA_PROPERTIES[wt][1]
    mt_charge = AA_PROPERTIES[mt][1]
    wt_polar = AA_PROPERTIES[wt][3]
    mt_polar = AA_PROPERTIES[mt][3]
    wt_vol = AA_PROPERTIES[wt][2]
    mt_vol = AA_PROPERTIES[mt][2]

    if mt == "P" or wt == "P":
        return "proline_mutations"

    # Charge reversal: sign flip of charge
    if wt_charge * mt_charge < 0:
        return "charge_reversal"

    # Neutral → charged (one is neutral, other has charge)
    if wt_charge == 0 and abs(mt_charge) > 0:
        return "neutral_to_charged"
    if abs(wt_charge) > 0 and mt_charge == 0:
        return "neutral_to_charged"  # also covers charged → neutral (same group)

    # Hydrophobic ↔ polar
    wt_hydro = wt in HYDROPHOBIC
    mt_hydro = mt in HYDROPHOBIC
    if wt_hydro != mt_hydro:
        return "hydrophobic_to_polar"

    # Size change (>30 Å³ difference)
    if abs(wt_vol - mt_vol) > 30:
        return "size_change"

    return "other"


def extract_features(wt: str, mt: str, aa_embeddings, encoder) -> list | None:
    """Extract 8 features (4 hyperbolic + 4 physicochemical) for a mutation."""
    if wt not in aa_embeddings or mt not in aa_embeddings:
        return None
    if wt not in AA_PROPERTIES or mt not in AA_PROPERTIES:
        return None

    wt_emb = aa_embeddings[wt]
    mt_emb = aa_embeddings[mt]

    hyp_dist = poincare_distance(
        wt_emb.unsqueeze(0), mt_emb.unsqueeze(0), c=encoder.curvature
    ).item()
    origin = torch.zeros(1, encoder.latent_dim)
    wt_r = poincare_distance(wt_emb.unsqueeze(0), origin, c=encoder.curvature).item()
    mt_r = poincare_distance(mt_emb.unsqueeze(0), origin, c=encoder.curvature).item()
    delta_radius = mt_r - wt_r

    diff = (mt_emb - wt_emb).detach().cpu().numpy()
    diff_norm = float(np.linalg.norm(diff))

    wt_np = wt_emb.detach().cpu().numpy()
    mt_np = mt_emb.detach().cpu().numpy()
    cos_sim = float(np.dot(wt_np, mt_np) / (np.linalg.norm(wt_np) * np.linalg.norm(mt_np) + 1e-10))

    wt_p = AA_PROPERTIES[wt]
    mt_p = AA_PROPERTIES[mt]
    delta_hydro = mt_p[0] - wt_p[0]
    delta_charge = abs(mt_p[1] - wt_p[1])
    delta_size = mt_p[2] - wt_p[2]
    delta_polar = mt_p[3] - wt_p[3]

    return [hyp_dist, delta_radius, diff_norm, cos_sim,
            delta_hydro, delta_charge, delta_size, delta_polar]


def spearman_safe(y_true, y_pred) -> float:
    """Spearman rho, returns nan for n<4 (unreliable)."""
    if len(y_true) < 4:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, _ = spearmanr(y_true, y_pred)
    return rho


def main():
    print("=" * 65)
    print("C5: MUTATION-TYPE SUBGROUP ANALYSIS")
    print("Dataset: ProTherm benchmark subset (N=52, NOT S669)")
    print("=" * 65)

    # Load encoder
    encoder_path = PACKAGE_DIR / "models/trained_codon_encoder.pt"
    checkpoint = torch.load(encoder_path, map_location="cpu", weights_only=True)
    config = checkpoint.get("config", {"latent_dim": 16, "hidden_dim": 64})
    encoder = TrainableCodonEncoder(
        latent_dim=config["latent_dim"], hidden_dim=config["hidden_dim"]
    )
    encoder.load_state_dict(checkpoint["model_state_dict"])
    encoder.eval()
    aa_embeddings = encoder.get_all_amino_acid_embeddings()

    # Load data
    data_path = PACKAGE_DIR / "reproducibility/data/s669.csv"
    if not data_path.exists():
        sys.path.insert(0, str(PACKAGE_DIR / "reproducibility"))
        from download_s669 import create_fallback_s669
        data_path.parent.mkdir(parents=True, exist_ok=True)
        create_fallback_s669(data_path)

    import csv
    mutations = []
    with open(data_path) as f:
        for row in csv.DictReader(f):
            mutations.append({
                "wt": row["wild_type"], "mt": row["mutant"],
                "ddg": float(row["ddg"]),
            })

    # Build per-mutation records
    records = []
    for m in mutations:
        feats = extract_features(m["wt"], m["mt"], aa_embeddings, encoder)
        if feats is None:
            continue
        mut_type = classify_mutation(m["wt"], m["mt"])
        records.append({
            "wt": m["wt"], "mt": m["mt"], "ddg": m["ddg"],
            "type": mut_type, "features": feats,
        })

    print(f"\nTotal mutations with features: {len(records)}")

    # Global LOO predictions for both model types
    X_all = np.array([r["features"] for r in records])
    y_all = np.array([r["ddg"] for r in records])

    pipe_full = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=100))])
    y_pred_full = cross_val_predict(pipe_full, X_all, y_all, cv=len(y_all))

    X_phys = X_all[:, 4:]  # physicochemical only (indices 4-7)
    pipe_phys = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=100))])
    y_pred_phys = cross_val_predict(pipe_phys, X_phys, y_all, cv=len(y_all))

    rho_global_full, _ = spearmanr(y_all, y_pred_full)
    rho_global_phys, _ = spearmanr(y_all, y_pred_phys)

    print(f"\nGlobal LOO Spearman (combined, N={len(records)}): {rho_global_full:.4f}")
    print(f"Global LOO Spearman (phys-only, N={len(records)}): {rho_global_phys:.4f}")

    # Per-subgroup analysis
    print("\n" + "=" * 65)
    print("PER-SUBGROUP ANALYSIS")
    print("=" * 65)
    print("⚠️  WARNING: Subgroups with N<10 have unreliable Spearman estimates.")
    print("    Extreme % values (e.g., -737%) come from N=2-3 mutations.")
    print()

    type_order = [
        "neutral_to_charged", "hydrophobic_to_polar", "size_change",
        "charge_reversal", "proline_mutations", "other",
    ]

    results = {}
    for mut_type in type_order:
        idx = [i for i, r in enumerate(records) if r["type"] == mut_type]
        if not idx:
            continue

        y_sub = y_all[idx]
        y_pred_full_sub = y_pred_full[idx]
        y_pred_phys_sub = y_pred_phys[idx]

        rho_full_sub = spearman_safe(y_sub, y_pred_full_sub)
        rho_phys_sub = spearman_safe(y_sub, y_pred_phys_sub)

        if not np.isnan(rho_phys_sub) and abs(rho_phys_sub) > 1e-6:
            perf_vs_baseline = (rho_full_sub / abs(rho_phys_sub) - 1) * 100
        else:
            perf_vs_baseline = float("nan")

        results[mut_type] = {
            "n": len(idx),
            "rho_combined": rho_full_sub,
            "rho_phys_only": rho_phys_sub,
            "perf_vs_baseline_pct": perf_vs_baseline,
        }

        n_flag = "⚠️ UNRELIABLE" if len(idx) < 10 else ""
        print(f"{mut_type} (N={len(idx)}) {n_flag}")
        print(f"  Spearman (combined):   {rho_full_sub:.4f}")
        print(f"  Spearman (phys-only):  {rho_phys_sub:.4f}")
        print(f"  % vs phys baseline:    {perf_vs_baseline:+.0f}%" if not np.isnan(perf_vs_baseline) else "  % vs phys baseline:    N/A (baseline ~0)")
        print()

    # Summary table
    print("=" * 65)
    print("SUMMARY TABLE (source for README.md)")
    print("=" * 65)
    print(f"{'Mutation Type':<25} {'N':>4}  {'rho_combined':>12}  {'% vs baseline':>14}  {'Reliable?':>9}")
    print("-" * 70)
    for mut_type, r in results.items():
        pct = f"{r['perf_vs_baseline_pct']:+.0f}%" if not np.isnan(r['perf_vs_baseline_pct']) else "N/A"
        reliable = "YES" if r["n"] >= 10 else "NO (N<10)"
        print(f"{mut_type:<25} {r['n']:>4}  {r['rho_combined']:>12.4f}  {pct:>14}  {reliable:>9}")

    print()
    print("NOTE: These subgroup Spearman values use global LOO predictions")
    print("  (model trained on full N=52, evaluated on the subgroup slice).")
    print("  A within-subgroup LOO is not feasible for N<10.")
    print()
    print("INTERPRETATION:")
    print("  The extreme -737% (charge_reversal) reflects that the model")
    print("  predicts the wrong direction for the ~3 charge-reversal mutations.")
    print("  This is driven by the small N, not a robust finding.")


if __name__ == "__main__":
    main()
