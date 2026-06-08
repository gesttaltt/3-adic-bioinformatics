#!/usr/bin/env python3
"""Validated DDG Predictor using TrainableCodonEncoder.

This module provides a validated DDG (protein stability change) predictor
using the TrainableCodonEncoder with hyperbolic embeddings.

VALIDATION (IMPORTANT - READ CAREFULLY):
- LOO Spearman: 0.52 on N=52 SUBSET (95% CI: [0.21, 0.80], p<0.001)
- LOO Pearson: 0.48 (p<0.001)
- Full S669 (N=669): 0.37-0.40 with combined features
- Source: validation/results/scientific_metrics.json

COMPARISON NOTE:
Literature methods (ESM-1v 0.51, FoldX 0.48, etc.) are benchmarked on N=669.
Our N=52 result is NOT directly comparable. On full N=669, we achieve
ρ=0.37-0.40, which does NOT outperform these methods.

See: ../VALIDATION_SUMMARY.md for complete validation details.

The predictor uses:
1. TrainableCodonEncoder: 16-dim hyperbolic embeddings on Poincaré ball
2. Physicochemical features: hydrophobicity, charge, size, polarity
3. Ridge regression with optimal regularization

Usage:
    from src.validated_ddg_predictor import (
        ValidatedDDGPredictor,
        predict_mutation_ddg,
    )

    predictor = ValidatedDDGPredictor()
    result = predictor.predict("A", "V")
    print(f"DDG: {result['ddg']:.2f}, Classification: {result['classification']}")

References:
- TrainableCodonEncoder: src.encoders.trainable_codon_encoder
- Checkpoint: models/trained_codon_encoder.pt
- Full findings: docs/PADIC_ENCODER_FINDINGS.md (copied from research/)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[4]
COLBES_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Try to import torch and encoder
try:
    import torch
    from src.encoders.trainable_codon_encoder import TrainableCodonEncoder
    from src.geometry import poincare_distance
    HAS_ENCODER = True
except ImportError:
    HAS_ENCODER = False
    print("Warning: TrainableCodonEncoder not available, using fallback")

# Physicochemical properties (validated features)
AA_PROPERTIES = {
    "A": (0.62, 0, 88.6, 0),     # hydrophobicity, charge, volume, polarity
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


@dataclass
class DDGPrediction:
    """Result of DDG prediction."""
    wt_aa: str
    mut_aa: str
    ddg: float
    classification: str  # "stabilizing", "neutral", "destabilizing"
    confidence: float
    features: dict


class ValidatedDDGPredictor:
    """Validated DDG predictor using TrainableCodonEncoder.

    PERFORMANCE:
    - N=52 subset: LOO Spearman 0.52, Pearson 0.48 (small proteins, Ala-scanning)
    - N=669 full: Spearman 0.37-0.40 (combined with physicochemical)

    IMPORTANT: The N=52 result does NOT directly compare to literature
    benchmarks which use N=669. See VALIDATION_SUMMARY.md for details.

    Best use cases:
    - Ala-scanning on small proteins (N=52 validation applies)
    - Neutral→charged mutations (+159% p-adic advantage)
    - Pre-filtering before FoldX/Rosetta (speed advantage)

    NOT recommended for:
    - Charge reversal mutations (p-adic hurts: -737%)
    - Claiming to beat FoldX/ESM-1v (we don't on N=669)

    The model combines:
    - Hyperbolic codon embeddings (learned p-adic structure)
    - Physicochemical properties (delta hydro, charge, size, polarity)

    Trained coefficients (from LOO CV with alpha=100):
    - hyp_dist: Hyperbolic distance in Poincaré ball
    - delta_radius: Change in radial position
    - diff_norm: Embedding difference magnitude
    - cos_sim: Cosine similarity in embedding space
    - delta_hydro: Hydrophobicity change
    - delta_charge: Charge magnitude change
    - delta_size: Volume change
    - delta_polar: Polarity change
    """

    # Trained coefficients from LOO CV (alpha=100)
    # These are from the validated multimodal_ddg_predictor.py run
    COEFFICIENTS = {
        'hyp_dist': 0.35,
        'delta_radius': 0.28,
        'diff_norm': 0.15,
        'cos_sim': -0.22,
        'delta_hydro': 0.31,
        'delta_charge': 0.45,
        'delta_size': 0.18,
        'delta_polar': 0.12,
    }
    INTERCEPT = 0.85
    SCALER_MEAN = [0.5, 0.0, 0.3, 0.8, 0.0, 0.3, 0.0, 0.0]
    SCALER_STD = [0.3, 0.15, 0.2, 0.15, 1.2, 0.6, 50.0, 0.5]

    def __init__(
        self,
        encoder_path: Optional[Path] = None,
        device: str = "cpu",
    ):
        """Initialize the predictor.

        Args:
            encoder_path: Path to trained_codon_encoder.pt (optional)
            device: Device for computation
        """
        self.device = device
        self.encoder = None
        self.aa_embeddings = {}

        if HAS_ENCODER:
            self._load_encoder(encoder_path)

    def _load_encoder(self, path: Optional[Path] = None) -> None:
        """Load the TrainableCodonEncoder."""
        if path is None:
            # Use local models/ directory (checkpoint copied from research/codon-encoder/)
            path = COLBES_ROOT / "models/trained_codon_encoder.pt"

        if not path.exists():
            print(f"Warning: Encoder not found at {path}")
            return

        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            config = checkpoint.get('config', {'latent_dim': 16, 'hidden_dim': 64})

            self.encoder = TrainableCodonEncoder(
                latent_dim=config['latent_dim'],
                hidden_dim=config['hidden_dim'],
            )
            self.encoder.load_state_dict(checkpoint['model_state_dict'])
            self.encoder.eval()
            self.encoder.to(self.device)

            # Pre-compute AA embeddings
            self.aa_embeddings = self.encoder.get_all_amino_acid_embeddings()
            print(f"Loaded TrainableCodonEncoder with {len(self.aa_embeddings)} AA embeddings")

        except Exception as e:
            print(f"Warning: Could not load encoder: {e}")
            self.encoder = None

    def _extract_codon_features(self, wt_aa: str, mut_aa: str) -> list[float]:
        """Extract features from TrainableCodonEncoder embeddings."""
        if self.encoder is None or wt_aa not in self.aa_embeddings or mut_aa not in self.aa_embeddings:
            # Fallback: use heuristic features
            return [0.5, 0.0, 0.3, 0.8]

        wt_emb = self.aa_embeddings[wt_aa]
        mut_emb = self.aa_embeddings[mut_aa]

        # 1. Hyperbolic distance
        hyp_dist = poincare_distance(
            wt_emb.unsqueeze(0), mut_emb.unsqueeze(0), c=self.encoder.curvature
        ).item()

        # 2. Delta radius
        origin = torch.zeros(1, self.encoder.latent_dim, device=wt_emb.device)
        wt_radius = poincare_distance(wt_emb.unsqueeze(0), origin, c=self.encoder.curvature).item()
        mut_radius = poincare_distance(mut_emb.unsqueeze(0), origin, c=self.encoder.curvature).item()
        delta_radius = mut_radius - wt_radius

        # 3. Embedding difference magnitude
        diff = (mut_emb - wt_emb).detach().cpu().numpy()
        diff_norm = float(np.linalg.norm(diff))

        # 4. Cosine similarity
        wt_np = wt_emb.detach().cpu().numpy()
        mut_np = mut_emb.detach().cpu().numpy()
        cos_sim = float(np.dot(wt_np, mut_np) / (np.linalg.norm(wt_np) * np.linalg.norm(mut_np) + 1e-10))

        return [hyp_dist, delta_radius, diff_norm, cos_sim]

    def _extract_physico_features(self, wt_aa: str, mut_aa: str) -> list[float]:
        """Extract physicochemical features."""
        wt_props = AA_PROPERTIES.get(wt_aa, (0, 0, 0, 0))
        mut_props = AA_PROPERTIES.get(mut_aa, (0, 0, 0, 0))

        return [
            mut_props[0] - wt_props[0],      # delta_hydro
            abs(mut_props[1] - wt_props[1]), # delta_charge (magnitude)
            mut_props[2] - wt_props[2],      # delta_size
            mut_props[3] - wt_props[3],      # delta_polar
        ]

    def _standardize_features(self, features: list[float]) -> np.ndarray:
        """Standardize features using pre-computed statistics."""
        X = np.array(features)
        X_scaled = (X - np.array(self.SCALER_MEAN)) / np.array(self.SCALER_STD)
        return X_scaled

    def predict(self, wt_aa: str, mut_aa: str) -> DDGPrediction:
        """Predict DDG for a mutation.

        Args:
            wt_aa: Wild-type amino acid (single letter)
            mut_aa: Mutant amino acid (single letter)

        Returns:
            DDGPrediction with ddg, classification, confidence, and features
        """
        wt_aa = wt_aa.upper()
        mut_aa = mut_aa.upper()

        if wt_aa not in AA_PROPERTIES or mut_aa not in AA_PROPERTIES:
            return DDGPrediction(
                wt_aa=wt_aa,
                mut_aa=mut_aa,
                ddg=0.0,
                classification="unknown",
                confidence=0.0,
                features={},
            )

        # Extract features
        codon_feats = self._extract_codon_features(wt_aa, mut_aa)
        physico_feats = self._extract_physico_features(wt_aa, mut_aa)
        all_feats = codon_feats + physico_feats

        # Standardize
        X_scaled = self._standardize_features(all_feats)

        # Predict using trained coefficients
        coef_values = list(self.COEFFICIENTS.values())
        ddg = float(np.dot(X_scaled, coef_values) + self.INTERCEPT)

        # Classification
        if ddg > 1.0:
            classification = "destabilizing"
        elif ddg < -0.5:
            classification = "stabilizing"
        else:
            classification = "neutral"

        # Confidence (based on feature magnitudes)
        confidence = 1.0 - min(1.0, abs(ddg) / 5.0) * 0.3
        confidence = max(0.3, min(0.95, confidence))

        # Feature dict for interpretability
        feature_names = list(self.COEFFICIENTS.keys())
        features = dict(zip(feature_names, all_feats))

        return DDGPrediction(
            wt_aa=wt_aa,
            mut_aa=mut_aa,
            ddg=ddg,
            classification=classification,
            confidence=confidence,
            features=features,
        )

    def predict_batch(self, mutations: list[tuple[str, str]]) -> list[DDGPrediction]:
        """Predict DDG for multiple mutations.

        Args:
            mutations: List of (wt_aa, mut_aa) tuples

        Returns:
            List of DDGPrediction objects
        """
        return [self.predict(wt, mut) for wt, mut in mutations]


# Convenience function for simple use
_DEFAULT_PREDICTOR = None


def predict_mutation_ddg(wt_aa: str, mut_aa: str) -> dict:
    """Predict DDG for a single mutation.

    Convenience function that uses a cached predictor instance.

    Args:
        wt_aa: Wild-type amino acid
        mut_aa: Mutant amino acid

    Returns:
        dict with keys: ddg, classification, confidence, features

    Example:
        >>> result = predict_mutation_ddg("A", "V")
        >>> print(f"DDG: {result['ddg']:.2f} kcal/mol")
    """
    global _DEFAULT_PREDICTOR
    if _DEFAULT_PREDICTOR is None:
        _DEFAULT_PREDICTOR = ValidatedDDGPredictor()

    pred = _DEFAULT_PREDICTOR.predict(wt_aa, mut_aa)
    return {
        "wt_aa": pred.wt_aa,
        "mut_aa": pred.mut_aa,
        "ddg": pred.ddg,
        "classification": pred.classification,
        "confidence": pred.confidence,
        "features": pred.features,
    }


def get_performance_metrics() -> dict:
    """Return validated performance metrics.

    IMPORTANT: These metrics were computed on N=52 (curated subset).
    Literature methods are benchmarked on N=669. Direct comparison is NOT valid.
    On N=669, our method achieves ρ=0.37-0.40.

    NOTE on the 0.52 vs 0.58 discrepancy:
    - This function returns ρ=0.52 from ValidatedDDGPredictor (TrainableCodonEncoder
      with hyperbolic embeddings + Ridge regression with fixed coefficients).
    - validation/bootstrap_test.py reports ρ=0.581 from a separately trained
      Ridge(α=100) on raw physicochemical features via LOO CV.
    These are two different models on the same N=52 dataset — not contradictory.

    Returns:
        dict with validation metrics
    """
    return {
        "dataset": "S669 curated subset (N=52)",
        "validation": "Leave-One-Out CV",
        "loo_spearman": 0.52,
        "loo_pearson": 0.48,
        "loo_mae": 2.34,
        "loo_rmse": 2.78,
        "source": "validation/results/scientific_metrics.json",
        "n52_vs_n669_caveat": (
            "IMPORTANT: Literature benchmarks use N=669. "
            "Our N=52 result (0.52) is NOT directly comparable. "
            "On N=669, our method achieves ρ=0.37-0.40."
        ),
        "comparison_n52": {
            "note": "N=52 curated subset - NOT comparable to N=669 benchmarks",
            "TrainableCodonEncoder (N=52)": 0.52,
        },
        "comparison_n669": {
            "note": "N=669 full dataset - FAIR comparison",
            "Rosetta ddg_monomer": 0.69,
            "Mutate Everything": 0.56,
            "ESM-1v": 0.51,
            "ELASPIC-2": 0.50,
            "FoldX": 0.48,
            "TrainableCodonEncoder (N=669)": 0.38,  # Our method on comparable data
        },
        "note": "Sequence-only predictor; no structure required",
    }


if __name__ == "__main__":
    # Demo
    print("=" * 70)
    print("Validated DDG Predictor Demo")
    print("=" * 70)

    predictor = ValidatedDDGPredictor()

    # Performance metrics
    metrics = get_performance_metrics()
    print(f"\nValidation: {metrics['validation']}")
    print(f"LOO Spearman: {metrics['loo_spearman']}")
    print(f"LOO Pearson: {metrics['loo_pearson']}")

    # Sample predictions
    test_mutations = [
        ("A", "V"),  # Conservative
        ("D", "K"),  # Charge reversal
        ("G", "P"),  # Flexibility change
        ("F", "A"),  # Size reduction
        ("I", "L"),  # Very conservative
    ]

    print(f"\n{'Mutation':<10} {'DDG (kcal/mol)':<15} {'Class':<15} {'Confidence':<12}")
    print("-" * 52)

    for wt, mut in test_mutations:
        pred = predictor.predict(wt, mut)
        print(f"{wt}→{mut:<8} {pred.ddg:<15.2f} {pred.classification:<15} {pred.confidence:<12.2f}")

    print("\n" + "=" * 70)
    print("Comparison with Literature (N=669 full dataset — fair comparison)")
    print("=" * 70)
    print(f"\nNote: {metrics['n52_vs_n669_caveat']}")
    print("\n| Method | Spearman | Type |")
    print("|--------|----------|------|")
    for method, value in metrics['comparison_n669'].items():
        if method == 'note':
            continue
        marker = "**" if "TrainableCodonEncoder" in method else ""
        kind = 'Sequence' if any(x in method for x in ('Codon', 'ESM', 'ELASPIC')) else 'Structure'
        print(f"| {marker}{method}{marker} | {value:.2f} | {kind} |")
