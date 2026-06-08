#!/usr/bin/env python3
"""
Experiment Runner: Optimizer Variants (#351-375)

Tests different optimizers and learning rate schedules.

From Research Plan:
- #351: SGD with momentum
- #352: Adam (current baseline)
- #353: AdamW (weight decay)
- #354: RAdam
- #355: NAdam
- #367: SWA (Stochastic Weight Averaging)
- #369: SAM (Sharpness-Aware Minimization)
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization optimizer.

    Seeks parameters that lie in flat minima (better generalization).
    """

    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        norm = torch.norm(
            torch.stack(
                [p.grad.norm(p=2) for group in self.param_groups for p in group["params"] if p.grad is not None]
            ),
            p=2,
        )
        return norm


class OptimizerExperimentRunner:
    """Runs optimizer experiments."""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device

    def load_stanford_raw(self, drug_class: str = "pi") -> tuple[pd.DataFrame, list[str], list[str]]:
        """Load Stanford HIVDB data."""
        data_dir = project_root / "data" / "research"

        file_mapping = {
            "pi": "stanford_hivdb_pi.txt",
            "nrti": "stanford_hivdb_nrti.txt",
            "nnrti": "stanford_hivdb_nnrti.txt",
            "ini": "stanford_hivdb_ini.txt",
        }

        drug_columns = {
            "pi": ["FPV", "ATV", "IDV", "LPV", "NFV", "SQV", "TPV", "DRV"],
            "nrti": ["ABC", "AZT", "D4T", "DDI", "FTC", "3TC", "TDF"],
            "nnrti": ["DOR", "EFV", "ETR", "NVP", "RPV"],
            "ini": ["BIC", "CAB", "DTG", "EVG", "RAL"],
        }

        filepath = data_dir / file_mapping[drug_class]
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        df = pd.read_csv(filepath, sep="\t", low_memory=False)
        prefix = "P"
        position_cols = [col for col in df.columns if col.startswith(prefix) and col[len(prefix) :].isdigit()]
        position_cols = sorted(position_cols, key=lambda x: int(x[len(prefix) :]))

        return df, position_cols, drug_columns[drug_class]

    def encode_amino_acids(self, df: pd.DataFrame, position_cols: list[str]) -> np.ndarray:
        """One-hot encode amino acid sequences."""
        aa_alphabet = "ACDEFGHIKLMNPQRSTVWY*-"
        aa_to_idx = {aa: i for i, aa in enumerate(aa_alphabet)}

        n_samples = len(df)
        n_positions = len(position_cols)
        n_aa = len(aa_alphabet)

        encoded = np.zeros((n_samples, n_positions * n_aa), dtype=np.float32)

        for idx, (_, row) in enumerate(df.iterrows()):
            for j, col in enumerate(position_cols):
                aa = str(row[col]).upper() if pd.notna(row[col]) else "-"
                if aa in aa_to_idx:
                    encoded[idx, j * n_aa + aa_to_idx[aa]] = 1.0
                else:
                    encoded[idx, j * n_aa + aa_to_idx["-"]] = 1.0

        return encoded

    def load_data(self, drug_class: str = "pi") -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Load HIV resistance data."""
        data = {}

        try:
            df, position_cols, drugs = self.load_stanford_raw(drug_class)
        except FileNotFoundError as e:
            print(f"  {e}")
            return data

        for drug in drugs:
            try:
                df_valid = df[df[drug].notna() & (df[drug] > 0)].copy()
                if len(df_valid) > 50:
                    X = self.encode_amino_acids(df_valid, position_cols)
                    y = np.log10(df_valid[drug].values + 1).astype(np.float32)
                    y = (y - y.min()) / (y.max() - y.min() + 1e-8)
                    data[drug] = (X, y)
                    print(f"  Loaded {drug}: {len(X)} samples")
            except Exception as e:
                print(f"  Could not load {drug}: {e}")

        return data

    def create_model(self, input_dim: int) -> nn.Module:
        """Create a predictor model."""
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ).to(self.device)

    def train_with_optimizer(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        optimizer_name: str,
        epochs: int = 100,
    ) -> dict:
        """Train model with specific optimizer."""
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(self.device)

        # Create optimizer
        lr = 1e-3
        if optimizer_name == "#351 SGD+Momentum":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        elif optimizer_name == "#352 Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == "#353 AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        elif optimizer_name == "#354 RAdam":
            optimizer = torch.optim.RAdam(model.parameters(), lr=lr)
        elif optimizer_name == "#355 NAdam":
            optimizer = torch.optim.NAdam(model.parameters(), lr=lr)
        elif optimizer_name == "#356 Lamb":
            # Use Adam as fallback (Lamb not in standard PyTorch)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == "#369 SAM":
            optimizer = SAM(model.parameters(), torch.optim.Adam, lr=lr, rho=0.05)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer if not isinstance(optimizer, SAM) else optimizer.base_optimizer, patience=10
        )

        best_corr = -1.0
        use_sam = isinstance(optimizer, SAM)

        for epoch in range(epochs):
            model.train()

            if use_sam:
                # SAM requires two forward passes
                pred = model(X_train_t).squeeze()
                loss = F.mse_loss(pred, y_train_t)

                # Add ranking loss
                if len(pred) > 1:
                    pred_mean = pred - pred.mean()
                    target_mean = y_train_t - y_train_t.mean()
                    cov = (pred_mean * target_mean).sum()
                    pred_std = torch.sqrt((pred_mean**2).sum() + 1e-8)
                    target_std = torch.sqrt((target_mean**2).sum() + 1e-8)
                    corr_loss = 1 - cov / (pred_std * target_std)
                    loss = loss + 0.5 * corr_loss

                loss.backward()
                optimizer.first_step(zero_grad=True)

                # Second forward pass
                pred = model(X_train_t).squeeze()
                loss2 = F.mse_loss(pred, y_train_t)
                if len(pred) > 1:
                    pred_mean = pred - pred.mean()
                    target_mean = y_train_t - y_train_t.mean()
                    cov = (pred_mean * target_mean).sum()
                    pred_std = torch.sqrt((pred_mean**2).sum() + 1e-8)
                    target_std = torch.sqrt((target_mean**2).sum() + 1e-8)
                    corr_loss = 1 - cov / (pred_std * target_std)
                    loss2 = loss2 + 0.5 * corr_loss
                loss2.backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.zero_grad()
                pred = model(X_train_t).squeeze()
                loss = F.mse_loss(pred, y_train_t)

                # Add ranking loss
                if len(pred) > 1:
                    pred_mean = pred - pred.mean()
                    target_mean = y_train_t - y_train_t.mean()
                    cov = (pred_mean * target_mean).sum()
                    pred_std = torch.sqrt((pred_mean**2).sum() + 1e-8)
                    target_std = torch.sqrt((target_mean**2).sum() + 1e-8)
                    corr_loss = 1 - cov / (pred_std * target_std)
                    loss = loss + 0.5 * corr_loss

                if torch.isnan(loss):
                    break

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step(loss.detach())

            # Evaluate
            if (epoch + 1) % 20 == 0:
                model.eval()
                with torch.no_grad():
                    test_pred = model(X_test_t).squeeze().cpu().numpy()
                    corr, _ = stats.spearmanr(test_pred, y_test)
                    if not np.isnan(corr):
                        best_corr = max(best_corr, corr)

        return {"optimizer": optimizer_name, "best_corr": best_corr}

    def run_experiment(self, drug_class: str = "pi") -> pd.DataFrame:
        """Run all optimizer experiments."""
        print(f"\n{'=' * 70}")
        print(f"OPTIMIZER EXPERIMENTS - {drug_class.upper()}")
        print(f"{'=' * 70}\n")

        data = self.load_data(drug_class)

        optimizers = [
            "#351 SGD+Momentum",
            "#352 Adam",
            "#353 AdamW",
            "#354 RAdam",
            "#355 NAdam",
            "#369 SAM",
        ]

        results = []

        for drug, (X, y) in data.items():
            print(f"\n--- Drug: {drug} ({len(X)} samples) ---")

            # Train/test split
            n = len(X)
            split_idx = int(0.8 * n)
            indices = np.random.permutation(n)
            train_idx, test_idx = indices[:split_idx], indices[split_idx:]

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            for opt_name in optimizers:
                print(f"  Testing {opt_name}...", end=" ")

                try:
                    model = self.create_model(X.shape[1])
                    result = self.train_with_optimizer(model, X_train, y_train, X_test, y_test, opt_name)
                    result["drug"] = drug
                    result["drug_class"] = drug_class
                    result["n_samples"] = len(X)
                    results.append(result)
                    print(f"corr = {result['best_corr']:+.3f}")
                except Exception as e:
                    print(f"FAILED: {e}")
                    results.append(
                        {
                            "drug": drug,
                            "drug_class": drug_class,
                            "optimizer": opt_name,
                            "best_corr": np.nan,
                            "error": str(e),
                        }
                    )

        return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Optimizer Experiments")
    parser.add_argument("--drug-class", type=str, default="pi", choices=["pi", "nrti", "nnrti", "ini", "all"])
    args = parser.parse_args()

    runner = OptimizerExperimentRunner()

    if args.drug_class == "all":
        all_results = []
        for dc in ["pi", "nrti", "nnrti", "ini"]:
            results = runner.run_experiment(dc)
            all_results.append(results)
        results = pd.concat(all_results, ignore_index=True)
    else:
        results = runner.run_experiment(args.drug_class)

    # Save results
    output_path = project_root / "results" / "optimizer_experiments.csv"
    results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY - Best Optimizer per Drug")
    print(f"{'=' * 70}\n")

    for drug in results["drug"].unique():
        drug_results = results[results["drug"] == drug].dropna(subset=["best_corr"])
        if len(drug_results) > 0:
            best = drug_results.loc[drug_results["best_corr"].idxmax()]
            print(f"{drug}: {best['optimizer']} -> {best['best_corr']:+.3f}")

    # Overall average
    print(f"\n{'=' * 70}")
    print("OVERALL - Average Correlation by Optimizer")
    print(f"{'=' * 70}\n")

    avg_by_opt = results.groupby("optimizer")["best_corr"].mean().sort_values(ascending=False)
    for opt, avg_corr in avg_by_opt.items():
        print(f"{opt}: {avg_corr:+.3f}")


if __name__ == "__main__":
    main()
