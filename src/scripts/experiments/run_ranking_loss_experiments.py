#!/usr/bin/env python3
"""
Experiment Runner: Ranking Loss Variants (#151-175)

Tests different ranking and correlation loss functions to improve
Spearman correlation performance.

From Research Plan:
- #151: Pearson correlation loss
- #152: Spearman correlation loss (differentiable)
- #153: Kendall tau loss
- #154: Margin ranking loss
- #155: Triplet ranking loss
- #156: Contrastive ranking loss
- #157: ListNet loss
- #158: ListMLE loss
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats


class RankingLosses:
    """Collection of ranking loss functions for experiments."""

    @staticmethod
    def pearson_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """#151: Pearson correlation loss."""
        pred_mean = pred - pred.mean()
        target_mean = target - target.mean()

        cov = (pred_mean * target_mean).sum()
        pred_std = torch.sqrt((pred_mean**2).sum() + 1e-8)
        target_std = torch.sqrt((target_mean**2).sum() + 1e-8)

        corr = cov / (pred_std * target_std)
        return 1 - corr

    @staticmethod
    def differentiable_spearman_loss(
        pred: torch.Tensor, target: torch.Tensor, temperature: float = 0.1
    ) -> torch.Tensor:
        """#152: Differentiable Spearman correlation loss using soft ranks."""
        pred.size(0)

        # Soft ranking using sigmoid comparisons
        pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)  # [n, n]
        soft_ranks_pred = torch.sigmoid(pred_diff / temperature).sum(dim=1)

        target_diff = target.unsqueeze(1) - target.unsqueeze(0)
        soft_ranks_target = torch.sigmoid(target_diff / temperature).sum(dim=1)

        # Pearson on soft ranks = Spearman
        return RankingLosses.pearson_loss(soft_ranks_pred, soft_ranks_target)

    @staticmethod
    def margin_ranking_loss(pred: torch.Tensor, target: torch.Tensor, margin: float = 0.1) -> torch.Tensor:
        """#154: Margin ranking loss - learns pairwise orderings."""
        n = pred.size(0)
        if n < 2:
            return torch.tensor(0.0, device=pred.device)

        # Create pairs
        idx1, idx2 = torch.triu_indices(n, n, offset=1)

        pred1, pred2 = pred[idx1], pred[idx2]
        target1, target2 = target[idx1], target[idx2]

        # Signs: 1 if target1 > target2, -1 otherwise
        signs = torch.sign(target1 - target2)

        # Margin ranking: max(0, -sign * (pred1 - pred2) + margin)
        loss = F.relu(-signs * (pred1 - pred2) + margin)
        return loss.mean()

    @staticmethod
    def triplet_ranking_loss(
        pred: torch.Tensor, target: torch.Tensor, margin: float = 0.1, n_triplets: int = 100
    ) -> torch.Tensor:
        """#155: Triplet ranking loss with anchor-positive-negative sampling."""
        n = pred.size(0)
        if n < 3:
            return torch.tensor(0.0, device=pred.device)

        # Sample triplets based on target ordering
        sorted_idx = torch.argsort(target)

        total_loss = 0.0
        count = 0

        for _ in range(min(n_triplets, n * (n - 1) * (n - 2) // 6)):
            # Sample anchor, positive (similar rank), negative (different rank)
            anchor_pos = torch.randint(1, n - 1, (1,)).item()
            anchor_idx = sorted_idx[anchor_pos]

            # Positive: one step away in rank
            pos_offset = torch.randint(0, 2, (1,)).item() * 2 - 1  # -1 or 1
            pos_idx = sorted_idx[max(0, min(n - 1, anchor_pos + pos_offset))]

            # Negative: far away in rank
            if anchor_pos < n // 2:
                neg_idx = sorted_idx[torch.randint(anchor_pos + n // 4, n, (1,)).item()]
            else:
                neg_idx = sorted_idx[torch.randint(0, anchor_pos - n // 4 + 1, (1,)).item()]

            # Triplet loss: d(anchor, positive) should be < d(anchor, negative)
            d_pos = torch.abs(pred[anchor_idx] - pred[pos_idx])
            d_neg = torch.abs(pred[anchor_idx] - pred[neg_idx])

            triplet_loss = F.relu(d_pos - d_neg + margin)
            total_loss += triplet_loss
            count += 1

        return total_loss / max(count, 1)

    @staticmethod
    def contrastive_ranking_loss(pred: torch.Tensor, target: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        """#156: Contrastive ranking loss - align similar, repel dissimilar."""
        n = pred.size(0)
        if n < 2:
            return torch.tensor(0.0, device=pred.device)

        # Normalize predictions
        pred_norm = (pred - pred.mean()) / (pred.std() + 1e-8)
        target_norm = (target - target.mean()) / (target.std() + 1e-8)

        # Similarity matrices
        pred_sim = torch.mm(pred_norm.unsqueeze(1), pred_norm.unsqueeze(0).T).squeeze()
        target_sim = torch.mm(target_norm.unsqueeze(1), target_norm.unsqueeze(0).T).squeeze()

        # Contrastive: predictions should have similar pairwise structure
        # Using MSE between similarity matrices
        return F.mse_loss(pred_sim, target_sim)

    @staticmethod
    def listnet_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """#157: ListNet loss - cross-entropy between probability distributions over rankings."""
        # Convert to probabilities via softmax
        pred_probs = F.softmax(pred, dim=0)
        target_probs = F.softmax(target, dim=0)

        # Cross-entropy
        return -torch.sum(target_probs * torch.log(pred_probs + 1e-8))

    @staticmethod
    def listmle_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """#158: ListMLE loss - maximum likelihood estimation for rankings."""
        n = pred.size(0)
        if n < 2:
            return torch.tensor(0.0, device=pred.device)

        # Sort by target (ground truth ordering)
        sorted_idx = torch.argsort(target, descending=True)
        sorted_pred = pred[sorted_idx]

        # ListMLE: -sum_i log(exp(s_i) / sum_j>=i exp(s_j))
        total_loss = 0.0
        for i in range(n):
            remaining = sorted_pred[i:]
            log_prob = sorted_pred[i] - torch.logsumexp(remaining, dim=0)
            total_loss -= log_prob

        return total_loss / n

    @staticmethod
    def ordinal_regression_loss(pred: torch.Tensor, target: torch.Tensor, n_levels: int = 5) -> torch.Tensor:
        """#168: Ordinal regression loss - treat as ordered classification."""
        # Discretize targets into levels
        target_min, target_max = target.min(), target.max()
        if target_max - target_min < 1e-8:
            return torch.tensor(0.0, device=pred.device)

        target_norm = (target - target_min) / (target_max - target_min)
        target_levels = (target_norm * (n_levels - 1)).long()

        # Pred should also predict levels
        pred_norm = (pred - target_min) / (target_max - target_min)
        pred_levels = pred_norm * (n_levels - 1)

        # Ordinal loss: for each threshold k, predict P(level > k)
        total_loss = 0.0
        for k in range(n_levels - 1):
            # Binary labels: 1 if level > k, 0 otherwise
            binary_target = (target_levels > k).float()
            # Predicted probability > threshold k
            binary_pred = torch.sigmoid(pred_levels - k - 0.5)
            total_loss += F.binary_cross_entropy(binary_pred, binary_target)

        return total_loss / (n_levels - 1)

    @staticmethod
    def soft_label_ranking_loss(pred: torch.Tensor, target: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
        """#171: Soft labels for ranking - Gaussian smoothed targets."""
        n = pred.size(0)

        # Create soft rankings using Gaussian kernels
        target.unsqueeze(1)  # [n, 1]
        positions = torch.arange(n, device=pred.device, dtype=pred.dtype).unsqueeze(0)  # [1, n]

        # Sort targets to get ranks
        sorted_idx = torch.argsort(target)
        ranks = torch.zeros_like(target)
        ranks[sorted_idx] = torch.arange(n, device=pred.device, dtype=pred.dtype)

        # Soft targets: Gaussian around true rank
        soft_targets = torch.exp(-((positions - ranks.unsqueeze(1)) ** 2) / (2 * sigma**2))
        soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True)

        # Soft predictions
        pred_sorted_idx = torch.argsort(pred)
        pred_ranks = torch.zeros_like(pred)
        pred_ranks[pred_sorted_idx] = torch.arange(n, device=pred.device, dtype=pred.dtype)

        soft_preds = torch.exp(-((positions - pred_ranks.unsqueeze(1)) ** 2) / (2 * sigma**2))
        soft_preds = soft_preds / soft_preds.sum(dim=1, keepdim=True)

        # KL divergence between soft distributions
        return F.kl_div(torch.log(soft_preds + 1e-8), soft_targets, reduction="batchmean")


class ExperimentRunner:
    """Runs ranking loss experiments systematically."""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.results = []

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

    def create_model(self, input_dim: int, latent_dim: int = 16) -> nn.Module:
        """Create a simple VAE-like encoder for experiments."""
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
            nn.Linear(64, 1),  # Predict resistance score
        ).to(self.device)

    def train_with_loss(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        loss_fn: callable,
        loss_name: str,
        epochs: int = 100,
        lr: float = 1e-3,
    ) -> dict:
        """Train model with specific loss function."""
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

        best_corr = -1.0

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            pred = model(X_train_t).squeeze()

            # MSE + ranking loss
            mse_loss = F.mse_loss(pred, y_train_t)
            rank_loss = loss_fn(pred, y_train_t)

            total_loss = mse_loss + 0.5 * rank_loss

            if torch.isnan(total_loss):
                print(f"    NaN at epoch {epoch}")
                break

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step(total_loss)

            # Evaluate
            if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
                model.eval()
                with torch.no_grad():
                    test_pred = model(X_test_t).squeeze().cpu().numpy()
                    corr, _ = stats.spearmanr(test_pred, y_test)
                    if not np.isnan(corr):
                        best_corr = max(best_corr, corr)

        return {"loss_name": loss_name, "best_corr": best_corr}

    def run_experiment(self, drug_class: str = "pi", drugs: list[str] | None = None) -> pd.DataFrame:
        """Run all ranking loss experiments."""
        print(f"\n{'=' * 70}")
        print(f"RANKING LOSS EXPERIMENTS - {drug_class.upper()}")
        print(f"{'=' * 70}\n")

        data = self.load_data(drug_class)
        if drugs:
            data = {k: v for k, v in data.items() if k in drugs}

        # Define loss functions to test
        loss_functions = {
            "#151 Pearson": RankingLosses.pearson_loss,
            "#152 DiffSpearman": RankingLosses.differentiable_spearman_loss,
            "#154 MarginRank": lambda p, t: RankingLosses.margin_ranking_loss(p, t, margin=0.1),
            "#155 Triplet": lambda p, t: RankingLosses.triplet_ranking_loss(p, t, margin=0.1),
            "#156 Contrastive": RankingLosses.contrastive_ranking_loss,
            "#157 ListNet": RankingLosses.listnet_loss,
            "#158 ListMLE": RankingLosses.listmle_loss,
            "#168 Ordinal": RankingLosses.ordinal_regression_loss,
            "#171 SoftLabel": RankingLosses.soft_label_ranking_loss,
        }

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

            for loss_name, loss_fn in loss_functions.items():
                print(f"  Testing {loss_name}...", end=" ")

                # Fresh model for each loss
                model = self.create_model(X.shape[1])

                try:
                    result = self.train_with_loss(
                        model, X_train, y_train, X_test, y_test, loss_fn, loss_name, epochs=100
                    )
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
                            "loss_name": loss_name,
                            "best_corr": np.nan,
                            "n_samples": len(X),
                            "error": str(e),
                        }
                    )

        return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Ranking Loss Experiments")
    parser.add_argument("--drug-class", type=str, default="pi", choices=["pi", "nrti", "nnrti", "ini", "all"])
    parser.add_argument("--drugs", type=str, nargs="+", default=None)
    args = parser.parse_args()

    runner = ExperimentRunner()

    if args.drug_class == "all":
        all_results = []
        for dc in ["pi", "nrti", "nnrti", "ini"]:
            results = runner.run_experiment(dc, args.drugs)
            all_results.append(results)
        results = pd.concat(all_results, ignore_index=True)
    else:
        results = runner.run_experiment(args.drug_class, args.drugs)

    # Save results
    output_path = project_root / "results" / "ranking_loss_experiments.csv"
    results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY - Best Loss Function per Drug")
    print(f"{'=' * 70}\n")

    for drug in results["drug"].unique():
        drug_results = results[results["drug"] == drug].dropna(subset=["best_corr"])
        if len(drug_results) > 0:
            best = drug_results.loc[drug_results["best_corr"].idxmax()]
            print(f"{drug}: {best['loss_name']} -> {best['best_corr']:+.3f}")

    # Overall best
    print(f"\n{'=' * 70}")
    print("OVERALL - Average Correlation by Loss Function")
    print(f"{'=' * 70}\n")

    avg_by_loss = results.groupby("loss_name")["best_corr"].mean().sort_values(ascending=False)
    for loss_name, avg_corr in avg_by_loss.items():
        print(f"{loss_name}: {avg_corr:+.3f}")


if __name__ == "__main__":
    main()
