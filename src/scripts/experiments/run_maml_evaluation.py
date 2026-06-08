"""MAML Few-Shot Evaluation for Drug Resistance Prediction.

Evaluates meta-learning performance on:
1. Held-out PI drugs (TPV, DRV - known to have fewer samples)
2. INI drugs (genuinely low-data: BIC=272, DTG=370)

Protocol:
- Meta-train on source drugs
- Meta-test on held-out drugs with few-shot adaptation
- Compare to fine-tuning baseline
"""

from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from models.maml_vae import MAMLVAE, MAMLConfig, compute_task_loss


@dataclass
class EvalConfig:
    """Configuration for MAML evaluation."""

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    meta_epochs: int = 100
    inner_lr: float = 0.01
    outer_lr: float = 0.001
    inner_steps: int = 5
    support_sizes: list[int] = field(default_factory=lambda: [5, 10, 20, 50, 100])
    n_eval_runs: int = 5  # Average over multiple runs


def load_drug_data(drug_class: str, drug: str) -> tuple[np.ndarray, np.ndarray]:
    """Load data for a specific drug."""
    data_dir = project_root / "data" / "research"

    file_mapping = {
        "pi": "stanford_hivdb_pi.txt",
        "nrti": "stanford_hivdb_nrti.txt",
        "nnrti": "stanford_hivdb_nnrti.txt",
        "ini": "stanford_hivdb_ini.txt",
    }

    filepath = data_dir / file_mapping[drug_class]
    df = pd.read_csv(filepath, sep="\t", low_memory=False)

    # Get position columns
    if drug_class == "pi":
        prefix = "P"
    elif drug_class in ["nrti", "nnrti"]:
        prefix = "RT"
    else:
        prefix = "IN"

    position_cols = [col for col in df.columns if col.startswith(prefix) and col[len(prefix) :].isdigit()]
    position_cols = sorted(position_cols, key=lambda x: int(x[len(prefix) :]))

    # Filter valid rows
    df_valid = df[df[drug].notna() & (df[drug] > 0)].copy()

    # Encode
    aa_alphabet = "ACDEFGHIKLMNPQRSTVWY*-"
    aa_to_idx = {aa: i for i, aa in enumerate(aa_alphabet)}

    n_samples = len(df_valid)
    n_positions = len(position_cols)
    n_aa = len(aa_alphabet)

    X = np.zeros((n_samples, n_positions * n_aa), dtype=np.float32)
    for idx, (_, row) in enumerate(df_valid.iterrows()):
        for j, col in enumerate(position_cols):
            aa = str(row[col]).upper() if pd.notna(row[col]) else "-"
            if aa in aa_to_idx:
                X[idx, j * n_aa + aa_to_idx[aa]] = 1.0
            else:
                X[idx, j * n_aa + aa_to_idx["-"]] = 1.0

    # Resistance values
    y = np.log10(df_valid[drug].values + 1).astype(np.float32)
    y = (y - y.min()) / (y.max() - y.min() + 1e-8)

    return X, y


class MetaTrainer:
    """MAML meta-trainer for drug resistance."""

    def __init__(self, cfg: MAMLConfig, eval_cfg: EvalConfig):
        self.cfg = cfg
        self.eval_cfg = eval_cfg
        self.device = torch.device(eval_cfg.device)
        self.model = MAMLVAE(cfg).to(self.device)

    def meta_train(self, task_data: dict[str, tuple[np.ndarray, np.ndarray]], epochs: int = 100):
        """Meta-train on multiple tasks (drugs)."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.eval_cfg.outer_lr)

        task_names = list(task_data.keys())
        print(f"\nMeta-training on {len(task_names)} tasks: {task_names}")

        for epoch in range(epochs):
            # Sample tasks for this meta-batch
            batch_tasks = np.random.choice(task_names, size=min(4, len(task_names)), replace=False)

            meta_loss = 0
            for task in batch_tasks:
                X, y = task_data[task]

                # Sample support and query sets
                n = len(X)
                idx = np.random.permutation(n)
                support_idx = idx[: n // 2]
                query_idx = idx[n // 2 :]

                support_x = torch.tensor(X[support_idx]).to(self.device)
                support_y = torch.tensor(y[support_idx]).to(self.device)
                query_x = torch.tensor(X[query_idx]).to(self.device)
                query_y = torch.tensor(y[query_idx]).to(self.device)

                # Inner loop: adapt to task
                adapted_model = self._inner_loop(support_x, support_y)

                # Compute loss on query set
                query_loss = compute_task_loss(self.cfg, adapted_model, query_x, query_y)
                meta_loss = meta_loss + query_loss

            # Outer loop: meta-update
            meta_loss = meta_loss / len(batch_tasks)
            optimizer.zero_grad()
            meta_loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch + 1}: meta_loss={meta_loss.item():.4f}")

    def _inner_loop(self, support_x: torch.Tensor, support_y: torch.Tensor) -> nn.Module:
        """Perform inner loop adaptation."""
        # Clone model for adaptation
        adapted_model = copy.deepcopy(self.model)

        # Adapt with gradient steps
        for _ in range(self.eval_cfg.inner_steps):
            loss = compute_task_loss(self.cfg, adapted_model, support_x, support_y)

            grads = torch.autograd.grad(loss, adapted_model.parameters(), create_graph=not self.cfg.first_order)

            # Update parameters
            for param, grad in zip(adapted_model.parameters(), grads, strict=False):
                param.data = param.data - self.eval_cfg.inner_lr * grad

        return adapted_model

    def evaluate_few_shot(
        self,
        X: np.ndarray,
        y: np.ndarray,
        support_size: int,
        n_runs: int = 5,
    ) -> tuple[float, float]:
        """Evaluate few-shot adaptation performance."""
        correlations = []

        for _ in range(n_runs):
            # Random split
            idx = np.random.permutation(len(X))
            support_idx = idx[:support_size]
            query_idx = idx[support_size:]

            support_x = torch.tensor(X[support_idx]).to(self.device)
            support_y = torch.tensor(y[support_idx]).to(self.device)
            query_x = torch.tensor(X[query_idx]).to(self.device)
            query_y = torch.tensor(y[query_idx]).to(self.device)

            # Adapt
            adapted_model = self._inner_loop(support_x, support_y)

            # Evaluate
            adapted_model.eval()
            with torch.no_grad():
                out = adapted_model(query_x)
                pred = out["prediction"].cpu().numpy()
                corr = np.corrcoef(pred, query_y.cpu().numpy())[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

        if correlations:
            return np.mean(correlations), np.std(correlations)
        return 0.0, 0.0


class FineTuningBaseline:
    """Fine-tuning baseline for comparison."""

    def __init__(self, cfg: MAMLConfig, eval_cfg: EvalConfig):
        self.cfg = cfg
        self.eval_cfg = eval_cfg
        self.device = torch.device(eval_cfg.device)

    def pretrain(self, task_data: dict[str, tuple[np.ndarray, np.ndarray]], epochs: int = 100):
        """Pretrain on all tasks (standard multi-task learning)."""
        self.model = MAMLVAE(self.cfg).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # Combine all data
        all_X = []
        all_y = []
        for _task, (X, y) in task_data.items():
            all_X.append(X)
            all_y.append(y)

        X_combined = np.vstack(all_X)
        y_combined = np.hstack(all_y)

        print(f"\nPre-training on {len(X_combined)} total samples")

        for epoch in range(epochs):
            # Random batch
            idx = np.random.choice(len(X_combined), size=min(64, len(X_combined)), replace=False)
            x = torch.tensor(X_combined[idx]).to(self.device)
            y = torch.tensor(y_combined[idx]).to(self.device)

            optimizer.zero_grad()
            loss = compute_task_loss(self.cfg, self.model, x, y)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch + 1}: loss={loss.item():.4f}")

    def evaluate_fine_tune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        support_size: int,
        n_runs: int = 5,
        fine_tune_epochs: int = 20,
    ) -> tuple[float, float]:
        """Evaluate fine-tuning performance."""
        correlations = []

        for _ in range(n_runs):
            # Random split
            idx = np.random.permutation(len(X))
            support_idx = idx[:support_size]
            query_idx = idx[support_size:]

            support_x = torch.tensor(X[support_idx]).to(self.device)
            support_y = torch.tensor(y[support_idx]).to(self.device)
            query_x = torch.tensor(X[query_idx]).to(self.device)
            query_y = torch.tensor(y[query_idx]).to(self.device)

            # Clone and fine-tune
            fine_tuned = copy.deepcopy(self.model)
            optimizer = torch.optim.Adam(fine_tuned.parameters(), lr=0.001)

            for _ in range(fine_tune_epochs):
                optimizer.zero_grad()
                loss = compute_task_loss(self.cfg, fine_tuned, support_x, support_y)
                loss.backward()
                optimizer.step()

            # Evaluate
            fine_tuned.eval()
            with torch.no_grad():
                out = fine_tuned(query_x)
                pred = out["prediction"].cpu().numpy()
                corr = np.corrcoef(pred, query_y.cpu().numpy())[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

        if correlations:
            return np.mean(correlations), np.std(correlations)
        return 0.0, 0.0


def run_pi_evaluation(eval_cfg: EvalConfig):
    """Evaluate on held-out PI drugs."""
    print("\n" + "=" * 80)
    print("PI DRUG EVALUATION")
    print("=" * 80)

    # Source drugs (meta-training)
    source_drugs = ["FPV", "ATV", "IDV", "LPV", "NFV", "SQV"]
    # Target drugs (few-shot evaluation)
    target_drugs = ["TPV", "DRV"]

    # Load source data
    source_data = {}
    input_dim = None
    for drug in source_drugs:
        try:
            X, y = load_drug_data("pi", drug)
            source_data[drug] = (X, y)
            input_dim = X.shape[1]
            print(f"  Loaded {drug}: {len(X)} samples")
        except Exception as e:
            print(f"  Error loading {drug}: {e}")

    if not source_data or input_dim is None:
        print("No source data available")
        return []

    # Create MAML config
    maml_cfg = MAMLConfig(
        input_dim=input_dim,
        latent_dim=16,
        inner_lr=eval_cfg.inner_lr,
        outer_lr=eval_cfg.outer_lr,
        inner_steps=eval_cfg.inner_steps,
    )

    # Train MAML
    print("\n--- Training MAML ---")
    maml_trainer = MetaTrainer(maml_cfg, eval_cfg)
    maml_trainer.meta_train(source_data, epochs=eval_cfg.meta_epochs)

    # Train baseline
    print("\n--- Training Fine-tuning Baseline ---")
    baseline = FineTuningBaseline(maml_cfg, eval_cfg)
    baseline.pretrain(source_data, epochs=eval_cfg.meta_epochs)

    # Evaluate on target drugs
    results = []
    for drug in target_drugs:
        print(f"\n--- Evaluating on {drug} ---")
        try:
            X, y = load_drug_data("pi", drug)
            print(f"  Total samples: {len(X)}")

            for n_support in eval_cfg.support_sizes:
                if n_support >= len(X) - 10:  # Need enough for query set
                    continue

                # MAML
                maml_corr, maml_std = maml_trainer.evaluate_few_shot(X, y, n_support, eval_cfg.n_eval_runs)

                # Fine-tuning
                ft_corr, ft_std = baseline.evaluate_fine_tune(X, y, n_support, eval_cfg.n_eval_runs)

                print(f"  n={n_support}: MAML={maml_corr:+.3f}±{maml_std:.3f}, FT={ft_corr:+.3f}±{ft_std:.3f}")

                results.append(
                    {
                        "drug_class": "pi",
                        "drug": drug,
                        "n_support": n_support,
                        "maml_corr": maml_corr,
                        "maml_std": maml_std,
                        "ft_corr": ft_corr,
                        "ft_std": ft_std,
                        "improvement": maml_corr - ft_corr,
                    }
                )

        except Exception as e:
            print(f"  Error: {e}")

    return results


def run_ini_evaluation(eval_cfg: EvalConfig):
    """Evaluate on INI drugs (genuinely low-data)."""
    print("\n" + "=" * 80)
    print("INI DRUG EVALUATION (LOW-DATA)")
    print("=" * 80)

    # All INI drugs
    all_drugs = ["BIC", "DTG", "EVG", "RAL"]

    # Load all INI data first
    all_data = {}
    input_dim = None
    for drug in all_drugs:
        try:
            X, y = load_drug_data("ini", drug)
            all_data[drug] = (X, y)
            input_dim = X.shape[1]
            print(f"  Loaded {drug}: {len(X)} samples")
        except Exception as e:
            print(f"  Error loading {drug}: {e}")

    if not all_data or input_dim is None:
        print("No INI data available")
        return []

    # Leave-one-out evaluation
    results = []

    for target_drug in all_data:
        print(f"\n--- Target: {target_drug} ---")

        # Source = all other drugs
        source_data = {d: data for d, data in all_data.items() if d != target_drug}

        # Create and train MAML
        maml_cfg = MAMLConfig(
            input_dim=input_dim,
            latent_dim=16,
            inner_lr=eval_cfg.inner_lr,
            outer_lr=eval_cfg.outer_lr,
            inner_steps=eval_cfg.inner_steps,
        )

        maml_trainer = MetaTrainer(maml_cfg, eval_cfg)
        maml_trainer.meta_train(source_data, epochs=eval_cfg.meta_epochs)

        baseline = FineTuningBaseline(maml_cfg, eval_cfg)
        baseline.pretrain(source_data, epochs=eval_cfg.meta_epochs)

        # Evaluate
        X, y = all_data[target_drug]
        for n_support in [5, 10, 20, 50]:
            if n_support >= len(X) - 10:
                continue

            maml_corr, maml_std = maml_trainer.evaluate_few_shot(X, y, n_support, eval_cfg.n_eval_runs)
            ft_corr, ft_std = baseline.evaluate_fine_tune(X, y, n_support, eval_cfg.n_eval_runs)

            print(f"  n={n_support}: MAML={maml_corr:+.3f}±{maml_std:.3f}, FT={ft_corr:+.3f}±{ft_std:.3f}")

            results.append(
                {
                    "drug_class": "ini",
                    "drug": target_drug,
                    "n_support": n_support,
                    "maml_corr": maml_corr,
                    "maml_std": maml_std,
                    "ft_corr": ft_corr,
                    "ft_std": ft_std,
                    "improvement": maml_corr - ft_corr,
                }
            )

    return results


def main():
    parser = argparse.ArgumentParser(description="MAML Few-Shot Evaluation")
    parser.add_argument("--meta-epochs", type=int, default=100)
    parser.add_argument("--inner-steps", type=int, default=5)
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--drug-class", type=str, default="all", choices=["pi", "ini", "all"])
    args = parser.parse_args()

    print("=" * 80)
    print("MAML FEW-SHOT EVALUATION FOR DRUG RESISTANCE")
    print("=" * 80)

    eval_cfg = EvalConfig(
        meta_epochs=args.meta_epochs,
        inner_steps=args.inner_steps,
        n_eval_runs=args.n_runs,
    )

    all_results = []

    if args.drug_class in ["pi", "all"]:
        pi_results = run_pi_evaluation(eval_cfg)
        all_results.extend(pi_results)

    if args.drug_class in ["ini", "all"]:
        ini_results = run_ini_evaluation(eval_cfg)
        all_results.extend(ini_results)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if all_results:
        print(f"\n{'Drug':<8} {'N':<6} {'MAML':<12} {'FT':<12} {'Δ':<8}")
        print("-" * 50)
        for r in all_results:
            delta = r["improvement"]
            delta_str = f"{delta:+.3f}" if delta > 0 else f"{delta:.3f}"
            print(
                f"{r['drug']:<8} {r['n_support']:<6} {r['maml_corr']:+.3f}±{r['maml_std']:.2f}  {r['ft_corr']:+.3f}±{r['ft_std']:.2f}  {delta_str}"
            )

        # Save
        results_df = pd.DataFrame(all_results)
        results_path = project_root / "results" / "maml_evaluation_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\nResults saved to: {results_path}")

        # Key findings
        maml_wins = sum(1 for r in all_results if r["improvement"] > 0)
        print(f"\nMAML outperforms fine-tuning in {maml_wins}/{len(all_results)} cases")


if __name__ == "__main__":
    main()
