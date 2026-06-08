#!/usr/bin/env python3
"""DDG Validation for PeptideVAE - Phase 4 (CRITICAL)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! IMPORTANT DISCLAIMER - DEPRECATED                                        !!
!!                                                                          !!
!! This validation script uses the NAIVE LSTM VAE and is DEPRECATED.        !!
!! For DDG prediction, use the Colbes p-adic approach directly:             !!
!!   - deliverables/partners/jose_colbes/reproducibility/                   !!
!!     train_padic_ddg_predictor_v2.py (validated ρ=0.585)                   !!
!!                                                                          !!
!! The existing TrainableCodonEncoder already provides proper p-adic        !!
!! embeddings that correlate with DDG.                                      !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Ground truth test: Must achieve Spearman ρ ≥ 0.585 (Colbes benchmark)

This validates that the learned embeddings capture stability information
by predicting ΔΔG (change in folding free energy) for mutations.

Method:
1. Load trained PeptideVAE
2. Embed amino acids using the encoder
3. Extract geometric features: hyp_dist, delta_radius
4. Train Ridge regression on S669 DDG data
5. Evaluate with Spearman correlation

Success criterion: ρ ≥ 0.585 (beats ESM-1v, ELASPIC-2, FoldX)

Usage:
    python src/scripts/peptide_vae/validate_ddg_phase4.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scipy.stats import pearsonr, spearmanr

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not available, cannot compute correlations")

try:
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("WARNING: sklearn not available, cannot train DDG predictor")

# Import PeptideVAE components
from scripts.peptide_vae.prediction_attempt_02 import (
    AA_PROPERTIES,
    AA_TO_IDX,
    AA_VOCAB,
    PAD_IDX,
    PeptideVAE,
    PeptideVAEConfig,
    poincare_distance,
)


def load_s669_data(filepath: Path) -> list[dict]:
    """Load S669 DDG benchmark dataset."""
    mutations = []
    with open(filepath) as f:
        lines = f.readlines()

    for line in lines[1:]:  # Skip header
        parts = line.strip().split(",")
        if len(parts) >= 6:
            try:
                mutations.append(
                    {
                        "pdb_id": parts[0],
                        "position": int(parts[2]),
                        "wild_type": parts[3].upper(),
                        "mutant": parts[4].upper(),
                        "ddg_experimental": float(parts[5]),
                    }
                )
            except (ValueError, IndexError):
                continue

    return mutations


def encode_amino_acid(model: PeptideVAE, aa: str, device: torch.device) -> torch.Tensor:
    """Encode a single amino acid using the VAE.

    We encode as a minimal peptide context to get meaningful embeddings.
    """
    # Create minimal context: AAA (target in middle)
    context = f"G{aa}G"  # Glycine flanks for neutral context

    # Encode
    encoded = [AA_TO_IDX.get(c, PAD_IDX) for c in context]
    padding = [PAD_IDX] * (50 - len(encoded))
    encoded = encoded + padding
    x = torch.tensor([encoded], dtype=torch.long, device=device)

    with torch.no_grad():
        mu, _ = model.encoder(x)
        z_hyp = model.reparameterize(mu, torch.zeros_like(mu))

    return z_hyp.squeeze(0)


def extract_ddg_features(
    model: PeptideVAE,
    mutations: list[dict],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract geometric features for DDG prediction.

    Features (from Colbes validation):
    - hyp_dist: Hyperbolic distance between wt and mut embeddings
    - delta_radius: Change in radial position (radius_mut - radius_wt)
    - delta_hydro: Change in hydrophobicity
    - delta_charge: Change in charge
    - delta_size: Change in size
    """
    model.eval()

    # Cache amino acid embeddings
    aa_embeddings = {}
    for aa in AA_VOCAB:
        aa_embeddings[aa] = encode_amino_acid(model, aa, device)

    features = []
    ddg_values = []

    for mut in mutations:
        wt_aa = mut["wild_type"]
        mut_aa = mut["mutant"]

        if wt_aa not in AA_VOCAB or mut_aa not in AA_VOCAB:
            continue

        z_wt = aa_embeddings[wt_aa]
        z_mut = aa_embeddings[mut_aa]

        # Geometric features (hyperbolic space)
        hyp_dist = poincare_distance(z_wt.unsqueeze(0), z_mut.unsqueeze(0), model.curvature).item()

        origin = torch.zeros_like(z_wt)
        radius_wt = poincare_distance(z_wt.unsqueeze(0), origin.unsqueeze(0), model.curvature).item()
        radius_mut = poincare_distance(z_mut.unsqueeze(0), origin.unsqueeze(0), model.curvature).item()
        delta_radius = radius_mut - radius_wt

        # Euclidean distance (for comparison)
        euc_dist = torch.norm(z_mut - z_wt).item()

        # Cosine similarity
        cos_sim = torch.cosine_similarity(z_wt.unsqueeze(0), z_mut.unsqueeze(0)).item()

        # Physicochemical features (from Colbes)
        wt_props = AA_PROPERTIES.get(wt_aa, {"hydro": 0, "charge": 0, "size": 0})
        mut_props = AA_PROPERTIES.get(mut_aa, {"hydro": 0, "charge": 0, "size": 0})

        delta_hydro = mut_props["hydro"] - wt_props["hydro"]
        delta_charge = abs(mut_props["charge"] - wt_props["charge"])
        delta_size = mut_props["size"] - wt_props["size"]

        # Combine features
        feature_vec = [
            hyp_dist,  # Hyperbolic distance (from trained VAE)
            delta_radius,  # Radial change
            euc_dist,  # Euclidean distance
            1 - cos_sim,  # Angular distance
            delta_hydro,  # Physicochemical
            delta_charge,
            delta_size,
        ]

        features.append(feature_vec)
        ddg_values.append(mut["ddg_experimental"])

    return np.array(features), np.array(ddg_values)


def train_and_evaluate_ddg(
    X: np.ndarray,
    y: np.ndarray,
) -> dict:
    """Train DDG predictor with Leave-One-Out cross-validation.

    This matches the Colbes benchmark methodology.
    """
    if not HAS_SKLEARN or not HAS_SCIPY:
        return {"error": "Missing scipy or sklearn"}

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Ridge regression (alpha=100 from Colbes)
    model = Ridge(alpha=100)

    # Leave-One-Out cross-validation (matches Colbes benchmark)
    loo = LeaveOneOut()
    y_pred = cross_val_predict(model, X_scaled, y, cv=loo)

    # Compute metrics
    spearman_r, spearman_p = spearmanr(y_pred, y)
    pearson_r, pearson_p = pearsonr(y_pred, y)

    mae = np.mean(np.abs(y_pred - y))
    rmse = np.sqrt(np.mean((y_pred - y) ** 2))

    # Feature importance (fit on full data)
    model.fit(X_scaled, y)
    feature_names = [
        "hyp_dist",
        "delta_radius",
        "euc_dist",
        "angular_dist",
        "delta_hydro",
        "delta_charge",
        "delta_size",
    ]
    importance = dict(zip(feature_names, model.coef_, strict=False))

    return {
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "mae": mae,
        "rmse": rmse,
        "n_samples": len(y),
        "feature_importance": importance,
    }


def main():
    print("=" * 70)
    print("Phase 4: DDG Validation (CRITICAL)")
    print("Success criterion: Spearman ρ ≥ 0.585 (Colbes benchmark)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load trained model
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "peptide_vae_attempt_02.pt"

    if not checkpoint_path.exists():
        print(f"\nERROR: Checkpoint not found at {checkpoint_path}")
        print("Please run prediction_attempt_02.py first")
        return 1

    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = PeptideVAEConfig()
    model = PeptideVAE(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded model from epoch {checkpoint['epoch']}")

    # Load S669 DDG data
    s669_path = PROJECT_ROOT / "deliverables" / "partners" / "jose_colbes" / "reproducibility" / "data" / "s669.csv"

    if not s669_path.exists():
        print(f"\nERROR: S669 data not found at {s669_path}")
        return 1

    print(f"\nLoading S669 data from: {s669_path}")
    mutations = load_s669_data(s669_path)
    print(f"Loaded {len(mutations)} mutations")

    # Extract features
    print("\nExtracting geometric features from VAE embeddings...")
    X, y = extract_ddg_features(model, mutations, device)
    print(f"Feature matrix: {X.shape}")
    print(f"DDG values: min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}")

    # Train and evaluate
    print("\nTraining DDG predictor with Leave-One-Out CV...")
    results = train_and_evaluate_ddg(X, y)

    # Display results
    print("\n" + "=" * 70)
    print("DDG PREDICTION RESULTS")
    print("=" * 70)

    print(f"\n  Spearman ρ: {results['spearman_r']:.4f} (p={results['spearman_p']:.2e})")
    print(f"  Pearson r:  {results['pearson_r']:.4f} (p={results['pearson_p']:.2e})")
    print(f"  MAE:        {results['mae']:.4f} kcal/mol")
    print(f"  RMSE:       {results['rmse']:.4f} kcal/mol")
    print(f"  N samples:  {results['n_samples']}")

    print("\n  Feature Importance:")
    for feat, coef in sorted(results["feature_importance"].items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"    {feat}: {coef:.4f}")

    # Comparison with literature
    print("\n" + "=" * 70)
    print("COMPARISON WITH LITERATURE")
    print("=" * 70)

    literature = {
        "Rosetta ddg_monomer (structure)": 0.69,
        "Our Target (Colbes validated)": 0.585,
        "Mutate Everything (sequence)": 0.56,
        "ESM-1v (sequence)": 0.51,
        "ELASPIC-2 (sequence)": 0.50,
        "FoldX (structure)": 0.48,
    }

    print("\n| Method | Spearman ρ |")
    print("|--------|------------|")
    for method, r in literature.items():
        marker = " <-- target" if "Target" in method else ""
        print(f"| {method} | {r:.2f}{marker} |")
    print(f"| **PeptideVAE (ours)** | **{results['spearman_r']:.2f}** |")

    # Validation verdict
    print("\n" + "=" * 70)
    print("PHASE 4 VERDICT")
    print("=" * 70)

    if results["spearman_r"] >= 0.585:
        print(f"\n✓ PHASE 4 PASSED: ρ = {results['spearman_r']:.3f} ≥ 0.585")
        print("  PeptideVAE embeddings capture stability information!")
        print("  Ready for production deployment.")
    elif results["spearman_r"] >= 0.50:
        print(f"\n~ PHASE 4 PARTIAL: ρ = {results['spearman_r']:.3f}")
        print("  Matches ESM-1v/ELASPIC-2 level but below target.")
        print("  Consider Phase 2 (radial hierarchy) training.")
    else:
        print(f"\n✗ PHASE 4 FAILED: ρ = {results['spearman_r']:.3f} < 0.50")
        print("  Embeddings don't capture stability information.")
        print("  Review architecture and training.")

    # Save results
    results_path = PROJECT_ROOT / "results" / "peptide_vae_ddg_validation.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON
    results_json = {
        "spearman_r": float(results["spearman_r"]),
        "spearman_p": float(results["spearman_p"]),
        "pearson_r": float(results["pearson_r"]),
        "pearson_p": float(results["pearson_p"]),
        "mae": float(results["mae"]),
        "rmse": float(results["rmse"]),
        "n_samples": results["n_samples"],
        "feature_importance": {k: float(v) for k, v in results["feature_importance"].items()},
        "passed": bool(results["spearman_r"] >= 0.585),
        "target": 0.585,
    }

    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    return 0 if results["spearman_r"] >= 0.585 else 1


if __name__ == "__main__":
    sys.exit(main())
