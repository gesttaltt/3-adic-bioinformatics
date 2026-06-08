#!/usr/bin/env python3
"""Bootstrap Validation for Carlos Brizuela AMP Activity Models.

This script validates the DRAMP-trained activity prediction models using:
1. Bootstrap resampling for confidence intervals
2. Cross-validation metrics (Pearson r, RMSE)
3. Comparison against random baseline

Usage:
    python validation/bootstrap_test.py
    python validation/bootstrap_test.py --n-bootstrap 500 --verbose
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Add shared module to path
_script_dir = Path(__file__).parent
_deliverables_dir = _script_dir.parent.parent.parent
sys.path.insert(0, str(_deliverables_dir))

# Add parent for loading DRAMP data
sys.path.insert(0, str(_script_dir.parent))

try:
    from scipy.stats import spearmanr, pearsonr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import joblib
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_score, cross_val_predict
    from sklearn.metrics import mean_squared_error
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class BootstrapResult:
    """Bootstrap validation results."""
    model_name: str
    n_samples: int
    n_bootstrap: int
    pearson_r: float
    pearson_p: float
    pearson_ci_lower: float
    pearson_ci_upper: float
    spearman_r: float
    spearman_p: float
    rmse: float
    rmse_ci_lower: float
    rmse_ci_upper: float
    random_baseline_r: float
    improvement_over_random: float


def load_curated_data():
    """Load curated AMP data from dramp_activity_loader."""
    from scripts.dramp_activity_loader import DRAMPLoader

    loader = DRAMPLoader()
    db = loader.generate_curated_database()
    return db


def bootstrap_correlation(y_true: np.ndarray, y_pred: np.ndarray,
                          n_bootstrap: int = 1000,
                          confidence: float = 0.95,
                          random_state: int = 42) -> dict:
    """Compute bootstrap confidence intervals for correlation.

    Args:
        y_true: True values
        y_pred: Predicted values
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default 0.95)
        random_state: Random seed for reproducibility
    """
    np.random.seed(random_state)  # Fix H1: Reproducibility
    n = len(y_true)
    correlations = []
    rmses = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        y_t = y_true[indices]
        y_p = y_pred[indices]

        # Compute metrics
        if HAS_SCIPY:
            r, _ = pearsonr(y_t, y_p)
            correlations.append(r)
        rmses.append(np.sqrt(np.mean((y_t - y_p) ** 2)))

    # Confidence intervals
    alpha = 1 - confidence
    r_ci = np.percentile(correlations, [alpha/2 * 100, (1 - alpha/2) * 100])
    rmse_ci = np.percentile(rmses, [alpha/2 * 100, (1 - alpha/2) * 100])

    return {
        'r_mean': np.mean(correlations),
        'r_std': np.std(correlations),
        'r_ci_lower': r_ci[0],
        'r_ci_upper': r_ci[1],
        'rmse_mean': np.mean(rmses),
        'rmse_std': np.std(rmses),
        'rmse_ci_lower': rmse_ci[0],
        'rmse_ci_upper': rmse_ci[1],
    }


def validate_model(model_path: Path, target: str = None,
                   n_bootstrap: int = 1000,
                   verbose: bool = False) -> Optional[BootstrapResult]:
    """Validate a trained activity model."""
    if not HAS_SKLEARN or not HAS_SCIPY:
        print("Error: scipy and sklearn required for validation")
        return None

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return None

    # Load model
    model_data = joblib.load(model_path)
    model = model_data['model']
    scaler = model_data.get('scaler')
    stored_metrics = model_data.get('metrics', {})

    # Load data
    db = load_curated_data()
    X, y = db.get_training_data(target)

    if len(X) < 10:
        print(f"Not enough data for {target}: {len(X)} samples")
        return None

    # Build a pipeline with scaler inside CV so each fold scales independently.
    # Using scaler.transform(X) before cross_val_predict leaks test-fold statistics
    # into the scaling step — the scaler was fitted on the full dataset including
    # samples that will later appear as test folds.
    if scaler is not None:
        cv_pipeline = Pipeline([('scaler', StandardScaler()), ('model', clone(model))])
    else:
        cv_pipeline = clone(model)

    # Cross-validation predictions (scaler refitted per fold, no leakage)
    n_folds = min(5, len(X))
    y_pred_cv = cross_val_predict(cv_pipeline, X, y, cv=n_folds)

    # Compute correlations
    pearson_r, pearson_p = pearsonr(y, y_pred_cv)
    spearman_r, spearman_p = spearmanr(y, y_pred_cv)
    rmse = np.sqrt(mean_squared_error(y, y_pred_cv))

    # Bootstrap confidence intervals
    bootstrap = bootstrap_correlation(y, y_pred_cv, n_bootstrap)

    # Random baseline (proper permutation test) - Fix H2
    np.random.seed(42)  # Reproducibility
    n_permutations = 1000  # Increased from 100 for statistical power
    perm_rs = []
    observed_r = abs(pearson_r)
    for _ in range(n_permutations):
        y_shuffled = np.random.permutation(y)
        r, _ = pearsonr(y_shuffled, y_pred_cv)  # Compare shuffled y vs predictions
        perm_rs.append(r)  # Keep sign, don't take abs (was inflating baseline)

    # Proper permutation p-value: proportion of permutations >= observed
    perm_p_value = np.mean(np.abs(perm_rs) >= observed_r)
    random_baseline = np.mean(np.abs(perm_rs))  # Report mean |r| under null

    result = BootstrapResult(
        model_name=model_path.stem,
        n_samples=len(X),
        n_bootstrap=n_bootstrap,
        pearson_r=pearson_r,
        pearson_p=pearson_p,
        pearson_ci_lower=bootstrap['r_ci_lower'],
        pearson_ci_upper=bootstrap['r_ci_upper'],
        spearman_r=spearman_r,
        spearman_p=spearman_p,
        rmse=rmse,
        rmse_ci_lower=bootstrap['rmse_ci_lower'],
        rmse_ci_upper=bootstrap['rmse_ci_upper'],
        random_baseline_r=random_baseline,
        improvement_over_random=pearson_r - random_baseline,
    )

    if verbose:
        print(f"\n{'-' * 50}")
        print(f"Model: {result.model_name}")
        print(f"  Samples: {result.n_samples}")
        print(f"  Pearson r: {result.pearson_r:.4f} (p={result.pearson_p:.4e})")
        print(f"  95% CI: [{result.pearson_ci_lower:.4f}, {result.pearson_ci_upper:.4f}]")
        print(f"  Spearman ρ: {result.spearman_r:.4f} (p={result.spearman_p:.4e})")
        print(f"  RMSE: {result.rmse:.4f} [{result.rmse_ci_lower:.4f}, {result.rmse_ci_upper:.4f}]")
        print(f"  Random baseline: {result.random_baseline_r:.4f}")
        print(f"  Improvement: {result.improvement_over_random:.4f}")

    return result


def main():
    """Run bootstrap validation for all models."""
    import argparse

    parser = argparse.ArgumentParser(description="Bootstrap Validation")
    parser.add_argument('--n-bootstrap', type=int, default=1000,
                        help='Number of bootstrap samples')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed output')
    parser.add_argument('--output', type=str,
                        default='validation/results/bootstrap_results.json',
                        help='Output file for results')
    args = parser.parse_args()

    print("=" * 60)
    print("CARLOS BRIZUELA DRAMP MODEL BOOTSTRAP VALIDATION")
    print("=" * 60)

    models_dir = Path(__file__).parent.parent / 'models'

    # Find all trained models
    model_files = list(models_dir.glob('activity_*.joblib'))

    if not model_files:
        print("No trained models found in models/")
        return 1

    print(f"\nFound {len(model_files)} trained models")
    print(f"Running {args.n_bootstrap} bootstrap samples per model...")

    results = []
    for model_path in sorted(model_files):
        # Extract target from filename (e.g., activity_staphylococcus.joblib)
        target = model_path.stem.replace('activity_', '')
        if target == 'general':
            target = None

        result = validate_model(model_path, target, args.n_bootstrap, args.verbose)
        if result:
            results.append(result)

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"{'Model':<25} {'N':>6} {'Pearson r':>10} {'95% CI':>20} {'p-value':>12}")
    print("-" * 75)

    all_significant = True
    for r in results:
        ci_str = f"[{r.pearson_ci_lower:.3f}, {r.pearson_ci_upper:.3f}]"
        sig = "***" if r.pearson_p < 0.001 else "**" if r.pearson_p < 0.01 else "*" if r.pearson_p < 0.05 else ""
        print(f"{r.model_name:<25} {r.n_samples:>6} {r.pearson_r:>10.4f} {ci_str:>20} {r.pearson_p:>10.4e} {sig}")
        if r.pearson_p >= 0.05:
            all_significant = False

    # Save results
    output_path = Path(__file__).parent.parent / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_dict = {
        'metadata': {
            'n_bootstrap': args.n_bootstrap,
            'validation_type': 'bootstrap_cross_validation',
            'timestamp': str(np.datetime64('now')),
        },
        'models': [
            {
                'name': r.model_name,
                'n_samples': r.n_samples,
                'pearson_r': r.pearson_r,
                'pearson_p': r.pearson_p,
                'pearson_ci': [r.pearson_ci_lower, r.pearson_ci_upper],
                'spearman_r': r.spearman_r,
                'spearman_p': r.spearman_p,
                'rmse': r.rmse,
                'rmse_ci': [r.rmse_ci_lower, r.rmse_ci_upper],
                'random_baseline': r.random_baseline_r,
                'improvement_over_random': r.improvement_over_random,
            }
            for r in results
        ],
        'summary': {
            'total_models': len(results),
            'significant_models': sum(1 for r in results if r.pearson_p < 0.05),
            'mean_pearson_r': np.mean([r.pearson_r for r in results]) if results else 0,
            'mean_rmse': np.mean([r.rmse for r in results]) if results else 0,
        }
    }

    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Final status
    if all_significant and results:
        print("\n✓ All models show statistically significant correlations (p < 0.05)")
        return 0
    else:
        print(f"\n✗ {len(results) - sum(1 for r in results if r.pearson_p < 0.05)} models not significant")
        return 1


if __name__ == '__main__':
    sys.exit(main())
